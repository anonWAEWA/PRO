import argparse
import json
from collections import defaultdict

import init_path
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import utils.data_utils as datu
import utils.misc_utils as mscu
import utils.model_utils as modu
from configs.clip import AVAILABLE_CLIP_MODELS, AVAILABLE_CLIP_PRETRAINED_DATASETS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataname", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=200)
    parser.add_argument(
        "-m",
        "--clip_model",
        type=str,
        required=False,
        default=modu.DEFAULT_CLIP_MODEL,
        choices=AVAILABLE_CLIP_MODELS,
    )
    parser.add_argument(
        "-p",
        "--clip_pretrained",
        type=str,
        required=False,
        default=modu.DEFAULT_CLIP_PRETRAINED,
        choices=AVAILABLE_CLIP_PRETRAINED_DATASETS,
    )
    args = parser.parse_args()
    return args


def main(args, folder, logger):
    # Get clip
    device = mscu.get_device()
    tokens = datu.get_tokens(args.dataname)
    model = modu.CLIP_Linear(
        tokens=tokens,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )

    loss_f = torch.nn.CrossEntropyLoss()

    # Get features
    saved_features = datu.get_saved_features(
        args.dataname, args.clip_model, args.clip_pretrained
    )
    if saved_features is not None:
        features = saved_features["features"]
        all_y = saved_features["y"]
    else:
        # Get data
        datasets = datu.get_datasets(
            args.dataname,
            train=True,
            test=True,
            transforms={"train": model.preprocess, "test": model.preprocess},
        )
        train_dataset, test_dataset = datasets["train"], datasets["test"]
        train_dataloader = DataLoader(train_dataset, args.batch_size)
        test_dataloader = DataLoader(test_dataset, args.batch_size)

        total_dataset = ConcatDataset([train_dataset, test_dataset])
        total_dataloader = DataLoader(total_dataset, args.batch_size)
        features = torch.zeros((0, model.feature_dim), device=model.device)
        all_y = torch.zeros(0, device=model.device, dtype=torch.int64)
        with torch.no_grad():
            for X, y in tqdm(total_dataloader, desc="Building features"):
                X, y = X.to(model.device), y.to(model.device)

                cur_feats = model.featurize(X)
                features = torch.cat((features, cur_feats), dim=0)
                all_y = torch.cat((all_y, y))
        datu.save_features(
            features=features,
            y=all_y,
            dataname=args.dataname,
            modelname=args.clip_model,
            pretrained_dataname=args.clip_pretrained,
        )

    # Get zero-shot performance
    metrics = defaultdict(list)
    model.evaluate_features(
        features, all_y, loss_f, metrics=metrics, metric_prefix="total", logger=logger,
    )

    with open(f"{folder}/results.json", "w") as f:
        json.dump(dict(metrics), f, indent=2)


if __name__ == "__main__":
    args = get_args()
    folder = (
        f"{init_path.ROOT_path}/results/zeroshot/{args.dataname}/"
        f"/model={args.clip_model}_pretrained={args.clip_pretrained}"
    )
    mscu.make_folder(folder)
    if mscu.is_completed(folder):
        print(f"Run existed in {folder}")
        exit()

    logger = mscu.get_logger(level="INFO", add_console=True, filename=f"{folder}/out.log")
    logger.info(f"Script Parameters: {json.dumps(vars(args), indent=2)}")
    main(args, folder, logger)

    mscu.note_completed(folder)
