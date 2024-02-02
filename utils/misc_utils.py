import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

PROJ_NAME = "test_time_adapt"
DATASETS = [
    "birdsnap",
    "caltech101",
    "cifar10",
    "cifar100",
    "country211",
    "dtd",
    "eurosat",
    "entity13",
    "entity30",
    "fer2013",
    "fgvcaircraft",
    "flowers102",
    "food101",
    "gtsrb",
    "imagenet",
    "mnist",
    "nonliving26",
    "oxfordpet",
    "officehome",
    "sst2",
    "stanfordcars",
    "stl10",
    "sun397",
    "living17",
    "pcam",
    "visda",
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_logger(level=None, filename=None, add_console=False):
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger(PROJ_NAME)

    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="w")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
    logger.propagate = False
    return logger


def make_folder(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)


def boolean(input):
    """Transform input to boolean, mostly used for ArgumentParser"""
    true_conditions = [
        lambda x: x.lower() == "true",
        lambda x: x.lower() == "yes",
        lambda x: x == "1",
    ]
    for cond in true_conditions:
        if cond(input) is True:
            return True
    return False


def type_or_none(input, type):
    if input.lower() == "none":
        return None
    else:
        return type(input)


def get_label_marginals(labels, num_labels=None):
    seen_labels, seen_counts = np.unique(labels, return_counts=True)
    seen_labels = seen_labels.astype(int)

    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    all_counts = np.zeros(num_labels)
    for idx, label in enumerate(seen_labels):
        all_counts[label] = seen_counts[idx]

    return all_counts / np.sum(all_counts)


def torch2np(array):
    if type(array) != np.ndarray:
        return array.cpu().numpy()
    return array


def get_kl_divergence(a, b, eps=1e-8):
    a, b = torch2np(a), torch2np(b)
    a += eps
    b += eps
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def get_tv_distance(a, b):
    a, b = torch2np(a), torch2np(b)
    return np.sum(np.abs(a - b))


def is_completed(folder):
    return os.path.exists(f"{folder}/done.txt")


def note_completed(folder):
    with open(f"{folder}/done.txt", "w") as f:
        print(f"done", file=f)


def sort_both(a, b, reverse=True):
    sorted_items = list(zip(*sorted(zip(a, b), reverse=reverse)))
    return sorted_items[0], sorted_items[1]


def get_zero_shot(dataname, parent_path, clip_model, clip_pretrained, logger=None):
    """Getting zero-shot results"""
    path = (
        f"{parent_path}/results/zeroshot/{dataname}"
        f"/model={clip_model}_pretrained={clip_pretrained}"
    )
    try:
        with open(f"{path}/results.json") as f:
            results = json.load(f)
        return results["total_accuracy"][0]
    except:
        if logger is not None:
            logger.warning(f"Failed getting zero-shot results for {dataname}")
        return None

def idx2onehot(label_idc, num_labels):
    onehot = torch.zeros((label_idc.shape[0], num_labels), device=label_idc.device)
    onehot[np.arange(label_idc.shape[0]), label_idc] = 1.0
    return onehot
