"""Adapted augmentation code"""
import random

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import utils.misc_utils as mscu


### Adapted from https://github.com/YBZh/Bridging_UDA_SSL
def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x_center = _sample_uniform(0, w)
    y_center = _sample_uniform(0, h)

    x0 = int(max(0, x_center - v / 2.0))
    y0 = int(max(0, y_center - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


FIX_MATCH_AUGMENTATION_POOL = [
    (AutoContrast, 0, 1),
    (Brightness, 0.05, 0.95),
    (Color, 0.05, 0.95),
    (Contrast, 0.05, 0.95),
    (Equalize, 0, 1),
    (Identity, 0, 1),
    (Posterize, 4, 8),
    (Rotate, -30, 30),
    (Sharpness, 0.05, 0.95),
    (ShearX, -0.3, 0.3),
    (ShearY, -0.3, 0.3),
    (Solarize, 0, 256),
    (TranslateX, -0.3, 0.3),
    (TranslateY, -0.3, 0.3),
]


def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()


class RandAugment:
    def __init__(self, n, augmentation_pool):
        assert n >= 1, "RandAugment N has to be a value greater than or equal to 1."
        self.n = n
        self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op(img, val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        img = Cutout(img, cutout_val)
        return img


# Supply transforms
class IdentityTransform(object):
    def __call__(self, img):
        return img


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)


def get_weak_transforms(end_transforms=None):
    """Get weak transform as specified in https://arxiv.org/pdf/2001.07685.pdf"""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(0, translate=(0.125, 0.125)),
            IdentityTransform() if end_transforms is None else end_transforms,
        ]
    )


def get_strong_transform(input_size, end_transforms=None, randaugment_n=2):
    """Adapted from https://github.com/YBZh/Bridging_UDA_SSL"""
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.convert("RGB")),  # Grayscale to RBG if needed
            transforms.RandomHorizontalFlip(),
            RandAugment(n=randaugment_n, augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,),
            IdentityTransform() if end_transforms is None else end_transforms,
        ]
    )


def get_must_weak_transform(input_size, mean, std):
    weak_transform = transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return weak_transform


def get_must_strong_transform(input_size,):
    strong_transform = create_transform(
        input_size=args.input_size,
        scale=(args.train_crop_min, 1),
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=args.image_mean,
        std=args.image_std,
    )
    return strong_transform


### Adapted from https://github.com/facebookresearch/simsiam/tree/main
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_simsiam_transform(end_transforms=None):
    """Adapted from https://github.com/facebookresearch/simsiam/tree/main"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            IdentityTransform() if end_transforms is None else end_transforms,
        ]
    )


def get_random_weak_aug(end_transforms=None, seed=42):
    mscu.set_seed(seed)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5  # not strengthened
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            IdentityTransform() if end_transforms is None else end_transforms,
        ]
    )
