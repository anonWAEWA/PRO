import logging
import os

import numpy as np
import pandas as pd
import sklearn  # Avoid lock in wilds
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_entity13,
    make_entity30,
    make_living17,
    make_nonliving26,
)
from robustness.tools.helpers import get_label_mapping
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

import utils.misc_utils as mscu
from configs.class_names import LABEL_NAMES
from configs.datasets import NUM_CLASSES
from configs.token_templates import TEMPLATES
from utils import ROOT_PATH

DATASET_PATH = "/path_to_data/"

CIFAR10_NAME = "cifar10"

logger = mscu.get_logger()


def save_features(
    features, y, dataname, modelname, pretrained_dataname, fpath=None, fname=None
):
    if fname is None:
        if fpath is None:
            fpath = f"{ROOT_PATH}/scratch/features/{modelname}_{pretrained_dataname}"
        mscu.make_folder(fpath)
        fname = f"{fpath}/{dataname}.npz"

    logger.info(f"Saving features to {fname}")
    np.savez(fname, features=features.cpu().numpy(), y=y.cpu().numpy())


def get_saved_features(
    dataname, modelname, pretrained_dataname, fpath=None, fname=None, device=None, logger=None,
):
    if fname is None:
        if fpath is None:
            fpath = f"{ROOT_PATH}/scratch/features/{modelname}_{pretrained_dataname}"
        fname = f"{fpath}/{dataname}.npz"

    if logger is None:
        logger = mscu.get_logger()
    logger.info(f"Reading from {fname}")

    if os.path.exists(fname):
        logger.info(f"Loading features from {fname}")
        saved_features = dict(np.load(fname))
        if device is None:
            device = mscu.get_device()
        saved_features["y"] = torch.from_numpy(saved_features["y"]).to(device)
        saved_features["features"] = torch.from_numpy(saved_features["features"]).to(
            device
        )
        return saved_features
    else:
        logger.info(f"Feature file {fname} does not exist")
        return None


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            transformed_img = self.transform(x[0])
            return transformed_img, x[1]
        else:
            return x

    @property
    def y_array(self):
        return self.dataset.y_array[self.indices]

    def __len__(self):
        return len(self.indices)


def split_idx(targets, num_classes, source_frac, seed):
    """
    Returns the indices of the source and target sets
    Input:
            dataset_len: length of the dataset
            source_frac: fraction of the dataset to use as source
            seed: seed for the random number generator
    Output:
            source_idx: indices of the source set
            target_idx: indices of the target set
    """

    np.random.seed(seed)
    idx_per_label = []
    for i in range(num_classes):
        idx_per_label.append(np.where(targets == i)[0])

    source_idx = []
    target_idx = []
    for i in range(num_classes):
        source_idx.extend(
            np.random.choice(
                idx_per_label[i], int(source_frac * len(idx_per_label[i])), replace=False
            )
        )
        target_idx.extend(np.setdiff1d(idx_per_label[i], source_idx, assume_unique=True))

    return np.array(source_idx), np.array(target_idx)


def dataset_with_targets(cls):
    """
    Modifies the dataset class to return target
    """

    def y_array(self):
        return np.array(self.targets).astype(int)

    dst_target = type(cls.__name__, (cls,), {"y_array": property(y_array)})
    return dst_target


def get_tokens(dataname):
    """Return list of list of tokens (num_tokens, num_templates).
    """
    label_names = LABEL_NAMES[dataname]
    if dataname in TEMPLATES:
        templates = TEMPLATES[dataname]
    else:
        logger.warning(f"Using default templates for tokens")
        templates = TEMPLATES["default"]
    tokens = [
        [template(label_name) for template in templates] for label_name in label_names
    ]
    return tokens


def initialize_transform(target_resolution=224):
    # default_mean = (0.4914, 0.4822, 0.4465)
    # default_std = (0.2023, 0.1994, 0.2010)

    # transform = T.Compose(
    #     [
    #         T.Resize(target_resolution),
    #         T.CenterCrop((target_resolution, target_resolution)),
    #         T.ToTensor(),
    #         T.Normalize(default_mean, default_std,),
    #     ]
    # )

    # Use openai's default transformation according to link:
    # https://github.com/openai/CLIP/blob/main/clip/clip.py

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    transform = T.Compose(
        [
            T.Resize(target_resolution, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(target_resolution),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            ),
        ]
    )

    return transform


def get_cifar10(
    train=True, test=False, transforms=None,
):
    root_dir = f"{DATASET_PATH}/{CIFAR10_NAME}"

    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    CIFAR10 = torchvision.datasets.CIFAR10
    if train:
        trainset = CIFAR10(
            root=root_dir, train=True, download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")
    if test:
        testset = CIFAR10(
            root=root_dir, train=False, download=True, transform=transforms["test"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_svhn(
    train=True, test=False, transforms=None,
):
    root_dir = f"{DATASET_PATH}/svhn"

    # Setup default transform
    if transforms is None:
        svhn_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((28, 28)),
            ]
        )

        transforms = {
            "train": svhn_transforms,
            "test": svhn_transforms,
        }
    elif not isinstance(transforms, dict):
        transforms = {
            "train": transforms,
            "test": transforms,
        }
    if train:
        trainset = torchvision.datasets.SVHN(
            root=root_dir, split="train", download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")
    if test:
        testset = torchvision.datasets.SVHN(
            root=root_dir, split="test", download=True, transform=transforms["test"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_eurosat(
    train=True, test=False, transforms=None, seed=42,
):
    root_dir = f"{DATASET_PATH}/eurosat"

    # Setup default transform
    if transforms is None:
        euroat_transform = initialize_transform(target_resolution=32)
        transforms = {
            "train": euroat_transform,
            "test": euroat_transform,
        }

    EuroSAT = dataset_with_targets(torchvision.datasets.EuroSAT)
    dataset = EuroSAT(root=root_dir, download=True, transform=None)

    train_idx, test_idx = split_idx(
        dataset.y_array, NUM_CLASSES["eurosat"], source_frac=0.8, seed=seed
    )

    if train:
        trainset = Subset(dataset, train_idx, transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = Subset(dataset, test_idx, transform=transforms["test"])
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_mnist(train=True, test=False, transforms=None):
    root_dir = f"{DATASET_PATH}/mnist"

    # Setup default transform
    if transforms is None:
        # mnist_transforms = initialize_transform(target_resolution=28)
        mnist_transforms = None
        transforms = {
            "train": mnist_transforms,
            "test": mnist_transforms,
        }

    MNIST = dataset_with_targets(torchvision.datasets.MNIST)
    if train:
        trainset = MNIST(
            root=root_dir, train=True, download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")
    if test:
        testset = MNIST(
            root=root_dir, train=False, download=False, transform=transforms["test"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_cifar100(train=True, test=False, transforms=None):
    root_dir = f"{DATASET_PATH}/cifar100"

    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    CIFAR100 = torchvision.datasets.CIFAR100

    if train:
        trainset = CIFAR100(
            root=root_dir, train=True, download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = CIFAR100(
            root=root_dir, train=False, download=True, transform=transforms["test"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


class CustomWTDataset(Dataset):
    def __init__(self, wt_dataset) -> None:
        super().__init__()
        self.wt_dataset = wt_dataset

    def __len__(self):
        return len(self.wt_dataset)

    def __getitem__(self, index):
        X, y, _ = self.wt_dataset[index]
        return X, y


def get_camelyon(train=True, test=False, transforms=None):
    dataset = Camelyon17Dataset(root_dir=DATASET_PATH)

    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    if train:
        trainset = dataset.get_subset("train", transform=transforms["train"])
        trainset = CustomWTDataset(trainset)
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = dataset.get_subset("val", transform=transforms["test"])
        valset = CustomWTDataset(valset)
        testset = dataset.get_subset("test", transform=transforms["test"])
        testset = CustomWTDataset(testset)

        testset = ConcatDataset([valset, testset])
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


class CustomFMoW(Dataset):
    def __init__(self, root_dir, train=True, transform=None) -> None:
        super().__init__()
        self.data_csv = pd.read_csv(f"{root_dir}/fmow_v1.1/rgb_metadata.csv")
        self.categories = self.data_csv["category"].unique()
        self.category_to_label = {
            self.categories[idx]: idx for idx in range(len(self.categories))
        }
        self.transform = transform

        if train:
            self.data = self.data_csv[self.data_csv["split"] == "train"]
        else:
            self.data = self.data_csv[self.data_csv["split"] != "train"]

        self.image_idc = self.data.index.tolist()
        self.image_paths = [
            f"{root_dir}/fmow_v1.1/images/rgb_img_{idx}.jpg" for idx in self.image_idc
        ]
        self.y_array = np.array(
            self.data["category"].apply(lambda cat: self.category_to_label[cat]).tolist()
        )

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_array[idx]

    def __len__(self):
        return len(self.data.index)


def get_fmow(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    # TODO: Ignoring unlabeled data for now. Decide whether this is a good choice later.
    # unlabeled_dataset = FMoWUnlabeledDataset(download=False, root_dir=root_dir)

    if train:
        trainset = CustomFMoW(
            root_dir=DATASET_PATH, train=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = CustomFMoW(
            root_dir=DATASET_PATH, train=False, transform=transforms["train"]
        )
        logger.info(f"Size of test data; {len(testset)}")
    # unlabeled_set = unlabeled_dataset.get_subset(
    #     "train_unlabeled", transform=None, load_y=True
    # )
    # union_dataset = ConcatDataset([trainset, valset, testset, unlabeled_set])
    # union_ydist = calculate_marginal(union_dataset.y_array, num_classes)

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_breeds(dataset=None, train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/imagenet/"

    if dataset == "living17":
        ret = make_living17(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "entity13":
        ret = make_entity13(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "entity30":
        ret = make_entity30(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "nonliving26":
        ret = make_nonliving26(f"{root_dir}/imagenet_hierarchy/", split="good")

    ImageFolder = folder.ImageFolder

    source_label_mapping = get_label_mapping("custom_imagenet", ret[1][0])
    target_label_mapping = get_label_mapping("custom_imagenet", ret[1][1])

    if train:
        trainset = ImageFolder(
            f"{root_dir}/imagenetv1/train/",
            label_mapping=source_label_mapping,
            transform=transforms["train"],
        )
        trainset.ret = ret
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = ImageFolder(
            f"{root_dir}/imagenetv1/val/",
            label_mapping=target_label_mapping,
            transform=transforms["test"],
        )
        testset.ret = ret
        logger.info(f"Size of test data; {len(testset)}")

        # TODO: Ignoring other target splits in RLSBench. Check if this is a good choice.
        # elif target_split == 2:
        #     targetset = ImageFolder(
        #         f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val",
        #         label_mapping=source_label_mapping,
        #     )

        #     target_train_idx, target_test_idx = split_idx(
        #         targetset.y_array, num_classes, source_frac=split_fraction, seed=seed
        #     )

        #     target_trainset = Subset(
        #         targetset, target_train_idx, transform=transforms["target_train"]
        #     )
        #     target_testset = Subset(
        #         targetset, target_test_idx, transform=transforms["target_test"]
        #     )

        # elif target_split == 3:
        #     targetset = ImageFolder(
        #         f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val",
        #         label_mapping=target_label_mapping,
        #     )

        #     target_train_idx, target_test_idx = split_idx(
        #         targetset.y_array, num_classes, source_frac=split_fraction, seed=seed
        #     )

        #     target_trainset = Subset(
        #         targetset, target_train_idx, transform=transforms["target_train"]
        #     )
        #     target_testset = Subset(
        #         targetset, target_test_idx, transform=transforms["target_test"]
        #     )

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


class EmptyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("EmptyDataset: index out of range")


def get_officehome(train=True, test=False, transforms=None, domain="Product"):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/officehome/"

    if train:
        trainset = ImageFolder(f"{root_dir}/{domain}/", transform=transforms["train"])
        # import pdb

        # pdb.set_trace()
        logger.info(f"Size of train data; {len(trainset)}")
    if test:
        testset = EmptyDataset()
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_visda(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/visda/"

    if train:
        trainset = ImageFolder(f"{root_dir}/train", transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = ImageFolder(f"{root_dir}/validation", transform=transforms["test"])
        testset = ImageFolder(f"{root_dir}/test", transform=transforms["test"])
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_food101(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/food101/"
    FOOD101 = torchvision.datasets.Food101

    if train:
        trainset = FOOD101(
            root=root_dir, split="train", download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = FOOD101(
            root=root_dir, split="test", download=True, transform=transforms["train"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_caltech101(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/caltech101/"
    CT101 = torchvision.datasets.Caltech101

    if train:
        trainset = CT101(root=root_dir, download=True, transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = EmptyDataset()
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_oxford_pet(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/oxfordpet/"
    DST = torchvision.datasets.OxfordIIITPet

    if train:
        trainset = DST(
            root=root_dir, split="trainval", download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = DST(
            root=root_dir, split="test", download=True, transform=transforms["test"]
        )
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_flowers102(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/flowers102/"
    DST = torchvision.datasets.Flowers102

    if train:
        trainset = DST(
            root=root_dir, split="train", download=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir, split="val", download=True, transform=transforms["test"]
        )
        testset = DST(
            root=root_dir, split="test", download=True, transform=transforms["test"]
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_voc2007(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/voc2007/"
    DST = torchvision.datasets.VOCDetection

    if train:
        trainset = DST(
            root=root_dir,
            year="2007",
            image_set="train",
            download=True,
            transform=transforms["train"],
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir,
            year="2007",
            image_set="val",
            download=True,
            transform=transforms["test"],
        )
        testset = DST(
            root=root_dir,
            year="2007",
            image_set="test",
            download=True,
            transform=transforms["test"],
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_stanfordcars(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/stanfordcars/"
    DST = torchvision.datasets.StanfordCars

    if train:
        trainset = DST(
            root=root_dir, split="train", download=True, transform=transforms["train"],
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        # TODO: Disabled for now b/c Stanford Cars links are unavailable.
        #   Followed instructions here: https://github.com/pytorch/vision/issues/7545
        #   But the test set is still unavailable.
        # testset = DST(
        #     root=root_dir, split="test", download=False, transform=transforms["test"],
        # )
        testset = EmptyDataset()

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_country211(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/country211/"

    if train:
        trainset = ImageFolder(f"{root_dir}/train", transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = ImageFolder(f"{root_dir}/valid", transform=transforms["test"])
        testset = ImageFolder(f"{root_dir}/test", transform=transforms["test"])
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_sst2(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/rendered-sst2/"

    if train:
        trainset = ImageFolder(f"{root_dir}/train", transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = ImageFolder(f"{root_dir}/valid", transform=transforms["test"])
        testset = ImageFolder(f"{root_dir}/test", transform=transforms["test"])
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_sun397(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/sun397/"
    DST = torchvision.datasets.SUN397

    if train:
        trainset = DST(root=root_dir, download=True, transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = EmptyDataset()

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_stl10(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/stl10/"
    DST = torchvision.datasets.STL10

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = DST(
            root=root_dir, download=True, split="test", transform=transforms["test"]
        )

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_fer2013(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/fer2013/"
    DST = torchvision.datasets.FER2013

    if train:
        trainset = DST(root=root_dir, split="train", transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        # Test set is not available b/c this is a challenge dataset.
        # testset = DST(root=root_dir, split="test", transform=transforms["test"])
        testset = EmptyDataset()

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_imagenet(train=True, test=False, transforms=None):
    from imagenetv2_pytorch import ImageNetV2Dataset

    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/imagenet/"

    if train:
        trainset = ImageNetV2Dataset(location=root_dir, transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        # Test set is not available b/c this is a challenge dataset.
        testset = EmptyDataset()
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_fgvcaircraft(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/fgvcaircraft/"
    DST = torchvision.datasets.FGVCAircraft

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir, download=True, split="val", transform=transforms["test"]
        )
        testset = DST(
            root=root_dir, download=True, split="test", transform=transforms["test"]
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_resisc45(train=True, test=False, transforms=None):
    from torchgeo.datasets import RESISC45

    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/resisc45/"
    DST = RESISC45

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transforms=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir, download=True, split="val", transforms=transforms["test"]
        )
        testset = DST(
            root=root_dir, download=True, split="test", transforms=transforms["test"]
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_dtd(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/dtd/"
    DST = torchvision.datasets.DTD

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir, download=True, split="val", transform=transforms["test"]
        )
        testset = DST(
            root=root_dir, download=True, split="test", transform=transforms["test"]
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_gtsrb(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/gtsrb/"
    DST = torchvision.datasets.GTSRB

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = DST(
            root=root_dir, download=True, split="test", transform=transforms["test"]
        )

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_pcam(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/pcam/"
    DST = torchvision.datasets.PCAM

    if train:
        trainset = DST(
            root=root_dir, download=True, split="train", transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        valset = DST(
            root=root_dir, download=True, split="val", transform=transforms["test"]
        )
        testset = DST(
            root=root_dir, download=True, split="test", transform=transforms["test"]
        )
        testset = ConcatDataset([valset, testset])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_kitti(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/kitti/"
    DST = torchvision.datasets.Kitti

    if train:
        trainset = DST(
            root=root_dir, download=True, train=True, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = DST(
            root=root_dir, download=True, train=False, transform=transforms["test"]
        )

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_birdsnap(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/birdsnap/download/images"

    if train:
        # Note /Mourning_Dove/181010.jpg is deleted b/c the downloaded image is corrupted
        trainset = ImageFolder(root_dir, transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = EmptyDataset()

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


def get_ucf101(train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    root_dir = f"{DATASET_PATH}/ucf101/"

    if train:
        trainset = ImageFolder(f"{root_dir}/train", transform=transforms["train"])
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = ImageFolder(f"{root_dir}/val", transform=transforms["test"])

        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


class CIFAR10_C:
    cifar_c = [
        "fog",
        "frost",
        "motion_blur",
        "brightness",
        "zoom_blur",
        "snow",
        "defocus_blur",
        "glass_blur",
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "speckle_noise",
        "spatter",
        "gaussian_blur",
        "saturate",
    ]
    severities = [1, 2, 3, 4, 5]

    def __init__(
        self,
        root=f"{DATASET_PATH}/cifar10c/",
        data_type=None,
        severity=1,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.transform = transform
        self.target_transform = target_transform

        assert data_type in self.cifar_c, f"{data_type} not from {self.cifar_c}"
        assert severity in self.severities, f"{severity} not from {self.severities}"

        data = np.load(root + "/" + data_type + ".npy")
        labels = np.load(root + "/" + "labels.npy")

        self.data = data[(severity - 1) * 10000 : (severity) * 10000]
        self.targets = labels[(severity - 1) * 10000 : (severity) * 10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)

    @property
    def y_array(self):
        return self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_cifar10_c(data_type, severity, train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    if train:
        trainset = CIFAR10_C(
            data_type=data_type, severity=severity, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = EmptyDataset()
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


class CIFAR100_C:
    cifar_c = [
        "fog",
        "frost",
        "motion_blur",
        "brightness",
        "zoom_blur",
        "snow",
        "defocus_blur",
        "glass_blur",
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "speckle_noise",
        "spatter",
        "gaussian_blur",
        "saturate",
    ]
    severities = [1, 2, 3, 4, 5]

    def __init__(
        self,
        root=f"{DATASET_PATH}/cifar100c/",
        data_type=None,
        severity=1,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.transform = transform
        self.target_transform = target_transform

        assert data_type in self.cifar_c, f"{data_type} not from {self.cifar_c}"
        assert severity in self.severities, f"{severity} not from {self.severities}"

        data = np.load(root + "/" + data_type + ".npy")
        labels = np.load(root + "/" + "labels.npy")

        self.data = data[(severity - 1) * 10000 : (severity) * 10000]
        self.targets = labels[(severity - 1) * 10000 : (severity) * 10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)

    @property
    def y_array(self):
        return self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_cifar100_c(data_type, severity, train=True, test=False, transforms=None):
    # Setup default transform
    if transforms is None:
        transforms = {
            "train": initialize_transform(),
            "test": initialize_transform(),
        }

    if train:
        trainset = CIFAR100_C(
            data_type=data_type, severity=severity, transform=transforms["train"]
        )
        logger.info(f"Size of train data; {len(trainset)}")

    if test:
        testset = EmptyDataset()
        logger.info(f"Size of test data; {len(testset)}")

    datasets = {}
    if train:
        datasets["train"] = trainset
    if test:
        datasets["test"] = testset

    return datasets


DATASET_GETTERS = {
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "svhn": get_svhn,
    "eurosat": get_eurosat,
    "mnist": get_mnist,
    "camelyon": get_camelyon,
    "fmow": get_fmow,
    "entity13": lambda **p: get_breeds(dataset="entity13", **p),
    "entity30": lambda **p: get_breeds(dataset="entity30", **p),
    "living17": lambda **p: get_breeds(dataset="living17", **p),
    "nonliving26": lambda **p: get_breeds(dataset="nonliving26", **p),
    "officehome": get_officehome,
    "visda": get_visda,
    "food101": get_food101,
    "caltech101": get_caltech101,
    "oxfordpet": get_oxford_pet,
    "flowers102": get_flowers102,
    "voc2007": get_voc2007,
    "stanfordcars": get_stanfordcars,
    "country211": get_country211,
    "sst2": get_sst2,
    "sun397": get_sun397,
    "stl10": get_stl10,
    "fer2013": get_fer2013,
    "imagenet": get_imagenet,
    "fgvcaircraft": get_fgvcaircraft,
    "resisc45": get_resisc45,
    "dtd": get_dtd,
    "gtsrb": get_gtsrb,
    "pcam": get_pcam,
    "kitti": get_kitti,
    "birdsnap": get_birdsnap,
    "officehome_product": lambda **p: get_officehome(domain="Product", **p),
    "officehome_art": lambda **p: get_officehome(domain="Art", **p),
    "officehome_clipart": lambda **p: get_officehome(domain="Clipart", **p),
    "officehome_realworld": lambda **p: get_officehome(domain="RealWorld", **p),
    "ucf101": get_ucf101,
    "cifar10c": get_cifar10_c,
    "cifar100c": get_cifar100_c,
}


def get_datasets(dataname, **params):
    if dataname not in DATASET_GETTERS:
        raise ValueError(f"Dataset {dataname} not supported")
    else:
        return DATASET_GETTERS[dataname](**params)


class DatasetWithIndices(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)
