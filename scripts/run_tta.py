import argparse
import copy
import json
from collections import defaultdict

import init_path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

import utils.adapt_utils as adpu
import utils.augmentation_utils as augu
import utils.data_utils as datu
import utils.misc_utils as mscu
import utils.model_utils as modu
from configs.clip import AVAILABLE_CLIP_MODELS, AVAILABLE_CLIP_PRETRAINED_DATASETS
from configs.datasets import NUM_CLASSES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataname",
        type=str,
        default="cifar10",
        choices=datu.DATASET_GETTERS.keys(),
    )
    parser.add_argument("-s", "--pseudolabel_src", type=str, default="PseudoLabel")
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
    parser.add_argument("--prior_estimate", type=str, default="zeroshot-soft")
    parser.add_argument("--tent_lambda", type=float, default=0.0)
    parser.add_argument("--iprior_tent_lambda", type=float, default=0.0)
    parser.add_argument("--pprior_tent_lambda", type=float, default=1.0)
    parser.add_argument("--memo_lambda", type=float, default=0.0)

    parser.add_argument("--marginal_smoothing", type=mscu.boolean, default=False)
    parser.add_argument(
        "--param_smoothing",
        type=lambda input: mscu.type_or_none(input, str),
        default=None,
    )
    parser.add_argument("--do_scheduling", type=mscu.boolean, default=False)
    parser.add_argument("--scheduler_type", type=str, default="linear")
    parser.add_argument("--do_teacher_student", type=mscu.boolean, default=False)
    parser.add_argument("--linear_bias", type=mscu.boolean, default=False)
    parser.add_argument(
        "--label_smoothing",
        type=lambda input: mscu.type_or_none(input, str),
        default=None,
    )
    parser.add_argument("--param_to_optimize", type=str, default="linear")
    parser.add_argument("--num_trained_prompts", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="ce")

    parser.add_argument("-c", "--confidence_threshold", type=float, default=0.9)
    parser.add_argument("-e", "--num_epoch", type=int, default=200)
    parser.add_argument(
        "--num_steps", type=lambda input: mscu.type_or_none(input, int), default=None
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=lambda input: mscu.type_or_none(input, int),
        default=None,
        help="If none, use size of entire dataset as batch size.",
    )
    parser.add_argument("--opt_type", type=str, default="sgd")
    parser.add_argument("-l", "--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0e-4)
    parser.add_argument("--momentum", type=float, default=1e-1)
    parser.add_argument("--warmup_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=4242)

    parser.add_argument("--check_pl_marginal", type=mscu.boolean, default=False)

    parser.add_argument(
        "--display_epoch", type=lambda input: mscu.type_or_none(input, int), default=1
    )
    parser.add_argument(
        "--display_steps", type=lambda input: mscu.type_or_none(input, int), default=None
    )
    parser.add_argument("--log_level", default="INFO", required=False)

    args = parser.parse_args()

    if args.prior_estimate == "teacher-soft":
        assert args.do_teacher_student
    return args


def main(args, folder, logger):
    # Get model
    device = mscu.get_device()
    tokens = datu.get_tokens(args.dataname)
    if args.param_to_optimize == "textual" and args.loss_type == "ce":
        model = modu.CLIP_Text(
            tokens=tokens,
            device=device,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            num_trained_prompts=args.num_trained_prompts,
        )
    elif args.loss_type == "cl":
        # TODO: Just use full clip for everything
        model = modu.CLIP_Full(
            tokens=tokens,
            device=device,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            num_trained_prompts=args.num_trained_prompts,
        )
    elif args.param_to_optimize == "linear" and args.loss_type == "ce":
        model = modu.CLIP_Linear(
            tokens=tokens,
            device=device,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            bias=args.linear_bias,
        )
    else:
        raise NotImplementedError()

    if args.do_teacher_student:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

    logger.info(f"Device: {device}")
    # logger.info(f"Clip tokens: {tokens}")

    # Get data
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
        total_dataset = ConcatDataset([train_dataset, test_dataset])

        total_dataloader = DataLoader(total_dataset, batch_size=100, shuffle=True)

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

    num_samples = features.shape[0]
    if args.batch_size is None:  # None means full batch
        args.batch_size = num_samples

    if args.opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            # lr=args.lr * args.batch_size / num_samples,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            # lr=args.lr * args.batch_size / num_samples,
            lr=args.lr,
            # momentum=args.momentum,
            # weight_decay=args.weight_decay,
        )
    if "visual" not in args.param_to_optimize:
        model.featurizer.lock_image_tower()

    loss_f = torch.nn.CrossEntropyLoss()

    if args.do_scheduling:
        # scheduler = adpu.FWScheduler(optimizer)
        # Code snippet from https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
        def warmup(current_step: int):
            if current_step < args.warmup_steps:  # current_step / warmup_steps * base_lr
                if args.scheduler_type == "linear":
                    scale = float((current_step + 1) / args.warmup_steps)
                elif args.scheduler_type == "gaussian":
                    T = current_step / args.warmup_steps
                    scale = torch.exp(torch.Tensor([-5 * (1 - T)])).item()
                else:
                    raise NotImplementedError()
                logger.info(f"Current step: {current_step}. LR cale {scale}")
                return scale
            else:
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    # Setup results dictionary
    metrics = defaultdict(list)
    metrics["tokens"] = tokens
    metrics.update(vars(args))

    if "visual" in args.param_to_optimize:
        datasets = datu.get_datasets(
            args.dataname,
            train=True,
            test=True,
            transforms={"train": model.preprocess, "test": model.preprocess},
        )
        train_dataset, test_dataset = datasets["train"], datasets["test"]
        total_dataset = ConcatDataset([train_dataset, test_dataset])
        total_dataloader = DataLoader(
            total_dataset, batch_size=args.batch_size, shuffle=True
        )
    elif args.pseudolabel_src == "FixMatch":
        # import pdb

        # pdb.set_trace()
        size = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        weak_tf, strong_tf = (
            augu.get_must_weak_transform(size, mean, std),
            augu.get_strong_transform(
                size,
                end_transforms=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean), std=torch.tensor(std)
                        ),
                    ]
                ),
            ),
            # model.preprocess,
        )
        tf = augu.MultipleTransforms([weak_tf, strong_tf])
        datasets = datu.get_datasets(
            args.dataname, train=True, test=True, transforms={"train": tf, "test": tf},
        )
        train_dataset, test_dataset = datasets["train"], datasets["test"]
        total_dataset = ConcatDataset([train_dataset, test_dataset])
        total_dataset = datu.DatasetWithIndices(total_dataset)
        total_dataloader = DataLoader(
            total_dataset, batch_size=args.batch_size, shuffle=True
        )
    else:
        total_dataset = TensorDataset(features, all_y, torch.arange(features.shape[0]))
        total_dataloader = DataLoader(
            total_dataset, batch_size=args.batch_size, shuffle=True
        )

    # Get label marginals
    num_classes = torch.max(all_y) + 1
    true_label_marginals = torch.from_numpy(mscu.get_label_marginals(all_y.cpu().numpy()))
    true_label_marginals = true_label_marginals.to(device)
    logger.info(f"True label marginals: {true_label_marginals}")

    if args.prior_estimate == "true":
        label_marginals = true_label_marginals
    elif args.prior_estimate == "uniform":
        label_marginals = torch.ones(num_classes, device=device) / num_classes
    elif args.prior_estimate in ("zeroshot-soft", "teacher-soft"):
        with torch.no_grad():
            logits = 100 * model.linear(features)
        label_marginals = logits.softmax(1).mean(0)
    else:
        raise NotImplementedError()

    def save_kl_and_tv(label_marg, verbose=False):
        kl_div = mscu.get_kl_divergence(true_label_marginals, label_marg)
        tv_dist = mscu.get_tv_distance(true_label_marginals, label_marg)
        metrics["kl_div"].append(kl_div)
        metrics["tv_dist"].append(tv_dist)
        if verbose:
            # logger.info(f"Estimated label prior: {label_marg}")
            logger.info(
                f"Estimated priro: KL Divergence: {kl_div}. TV Distance: {tv_dist}"
            )

    save_kl_and_tv(label_marginals, verbose=True)

    pseudolabel_counts = torch.zeros((all_y.shape[0], num_classes), device=device)
    pseudolabel_need_ini = torch.ones(all_y.shape[0], device=device, dtype=bool)
    pseudolabel_probs = torch.zeros((all_y.shape[0], num_classes), device=device)
    fw_log_prob = None
    fw_step = 0
    logits_fw_lr = (4 / (torch.arange(args.num_epoch + 1, device=device) + 7)) ** (2 / 3)
    pl_fw_lr = 9 / (torch.arange(args.num_epoch + 1, device=device) + 8)
    moving_population_marginal = torch.clone(label_marginals)

    model.evaluate_features(
        features,
        all_y,
        loss_f,
        metrics=metrics,
        metric_prefix="Before TTA",
        logger=logger,
    )
    total_step = 0
    for epoch in range(args.num_epoch):
        step = 0

        for data in tqdm(total_dataloader, desc=f"Epoch {epoch}"):
            if args.pseudolabel_src != "FixMatch":
                X, _, batch_indices = data
                batch_indices = batch_indices.to(device)
                X = X.to(device)
            else:
                X_weak, X_strong = data[0]
                X_weak, X_strong = X_weak.to(device), X_strong.to(device)

                batch_indices = data[2]
                batch_indices = batch_indices.to(device)

            # GD
            optimizer.zero_grad()

            if (
                args.pseudolabel_src == "PseudoLabel"
                and args.param_to_optimize == "textual"
            ):
                logits = model.forward_image_features(X, train=True)
            elif (
                args.pseudolabel_src == "PseudoLabel"
                and args.param_to_optimize == "linear"
            ):
                logits = 100 * model.linear(X)
            elif (
                args.pseudolabel_src == "PseudoLabel"
                and args.param_to_optimize == "visual+textual"
            ):
                image_features, text_features = model.get_features(X, train=True)
                logits = 100.0 * image_features @ text_features.T
            elif (
                args.pseudolabel_src == "FixMatch" and args.param_to_optimize == "linear"
            ):
                with torch.no_grad():
                    image_features_weak = model.featurize(X_weak)
                    image_features_strong = model.featurize(X_strong)
                    logits_weak = 100.0 * model.linear(image_features_weak)
                logits = 100.0 * model.linear(image_features_strong)
            else:
                raise NotImplementedError()

            if args.do_teacher_student:
                adpu.update_teacher_with_student(teacher_model, model, decay=0.999)
                # adpu.update_teacher_with_student(teacher_model, model, decay=0.99)
                with torch.no_grad():
                    if (
                        args.pseudolabel_src == "PseudoLabel"
                        and args.param_to_optimize == "textual"
                    ):
                        teacher_logits = teacher_model.forward_image_features(
                            X, train=True
                        )
                    elif (
                        args.pseudolabel_src == "PseudoLabel"
                        and args.param_to_optimize == "linear"
                    ):
                        teacher_logits = 100.0 * teacher_model.linear(X)
                    elif (
                        args.pseudolabel_src == "FixMatch"
                        and args.param_to_optimize == "linear"
                    ):
                        teacher_logits = 100.0 * teacher_model.linear(image_features_weak)
                    else:
                        raise NotImplementedError()
                pseudolabeling_logits = teacher_logits

                # Update prior
                if args.prior_estimate == "teacher-soft":
                    label_marginals = teacher_logits.softmax(1).mean(0)
                    save_kl_and_tv(label_marginals, verbose=True)
            else:
                if args.pseudolabel_src == "PseudoLabel":
                    pseudolabeling_logits = logits
                else:
                    pseudolabeling_logits = logits_weak

            if args.label_smoothing is None or args.label_smoothing.split("_")[0] != "fw":
                # Get Pseudolabels
                _, pseudo_y, pseudo_frac, mask = adpu.get_pseudo_labels(
                    pseudolabeling_logits,
                    confidence_threshold=args.confidence_threshold,
                    return_mask=True,
                )
                metrics["pseudo_fracs"].append(pseudo_frac.detach().cpu().item())

                if args.check_pl_marginal:
                    pseudo_marginal = mscu.get_label_marginals(pseudo_y.cpu().numpy())
                    logger.info(f"Marginal of pseudolabels:\n{pseudo_marginal}")
            else:
                # Get FW Pseudolabels
                cur_log_prob = torch.clone(pseudolabeling_logits.log_softmax(-1)).detach()
                if fw_log_prob is None:
                    fw_log_prob = cur_log_prob
                else:
                    cur_logits_lr = logits_fw_lr[fw_step]
                    fw_log_prob = (
                        1 - cur_logits_lr
                    ) * fw_log_prob + cur_logits_lr * cur_log_prob
                _, pseudo_y, pseudo_frac, mask = adpu.get_pseudo_labels_from_logprob(
                    fw_log_prob,
                    confidence_threshold=args.confidence_threshold,
                    return_mask=True,
                )
                fw_step += 1

            if args.loss_type == "ce":
                if args.label_smoothing is None:
                    loss = loss_f(logits[mask], pseudo_y) * pseudo_frac
                elif args.label_smoothing == "avg":
                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    pseudolabel_counts.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    cur_pseudolabels = pseudolabel_counts[batch_indices, :]
                    avg_pseudolabels = cur_pseudolabels[mask] / torch.sum(
                        cur_pseudolabels[mask], axis=1, keepdim=True
                    )

                    # Record whether pseudolabels are getting closer to true
                    pseudolabel_ce = loss_f(
                        torch.log(avg_pseudolabels + 1e-8), all_y[batch_indices][mask],
                    )
                    metrics["pseudolabels_ce"].append(pseudolabel_ce.cpu().item())

                    loss = loss_f(logits[mask], avg_pseudolabels) * pseudo_frac
                elif args.label_smoothing == "ema":
                    new_pseudolabels = torch.zeros(
                        (all_y.shape[0], num_classes), device=device
                    )

                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    new_pseudolabels.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    pl_ema = 0.01
                    pseudolabel_probs = (
                        1 - pl_ema
                    ) * pseudolabel_probs + pl_ema * new_pseudolabels

                    # Compute loss
                    cur_pseudolabels = pseudolabel_probs[batch_indices, :][mask]
                    loss = loss_f(logits[mask], cur_pseudolabels) * pseudo_frac
                elif args.label_smoothing.split("_")[0] == "ema-warm":
                    new_pseudolabels = torch.zeros(
                        (all_y.shape[0], num_classes), device=device
                    )

                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    new_pseudolabels.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    pl_ema = 0.01
                    pseudolabel_probs = (
                        1 - pl_ema
                    ) * pseudolabel_probs + pl_ema * new_pseudolabels

                    warmup_strat = (
                        args.label_smoothing.split("_")[1]
                        if len(args.label_smoothing.split("_")) > 1
                        else "simple"
                    )

                    if warmup_strat == "simple":
                        if epoch > 80:
                            pseudolabel_probs = pseudolabel_probs / torch.sum(
                                pseudolabel_probs, axis=1, keepdim=True
                            )
                    elif warmup_strat == "gaussian":
                        if epoch > 80:
                            scale = 1.0
                        else:
                            T = epoch / 80
                            scale = torch.exp(torch.Tensor([-5 * (1 - T)])).to(device)
                        pseudolabel_probs = (
                            pseudolabel_probs
                            / torch.sum(pseudolabel_probs, axis=1, keepdim=True)
                            * scale
                        )
                        print(f"{scale=}")
                        print(f"{torch.sum(pseudolabel_probs, axis=1)=}")

                    # Compute loss
                    cur_pseudolabels = pseudolabel_probs[batch_indices, :][mask]
                    loss = loss_f(logits[mask], cur_pseudolabels) * pseudo_frac
                elif args.label_smoothing == "temp":
                    new_pseudolabels = torch.zeros(
                        (all_y.shape[0], num_classes), device=device
                    )

                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    new_pseudolabels.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    pl_decay = 0.6
                    pseudolabel_probs = (pl_decay) * pseudolabel_probs + (
                        1 - pl_decay
                    ) * new_pseudolabels
                    pseudolabel_target = pseudolabel_probs / (1 - pl_decay ** (epoch + 1))

                    if epoch > 80:
                        loss_scale = 1.0
                    else:
                        T = epoch / 80
                        loss_scale = torch.exp(torch.Tensor([-5 * (1 - T)])).to(device)
                    loss_scale = loss_scale / pseudolabel_probs.shape[1]

                    # Compute loss
                    cur_pseudolabels = pseudolabel_target[batch_indices, :][mask]
                    loss = (
                        loss_f(logits[mask], cur_pseudolabels) * pseudo_frac * loss_scale
                    )
                elif args.label_smoothing == "ema-hardini":
                    new_pseudolabels = torch.zeros(
                        (all_y.shape[0], num_classes), device=device
                    )

                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    new_pseudolabels.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    pseudolabel_probs[pseudolabel_need_ini] = new_pseudolabels[
                        pseudolabel_need_ini
                    ]
                    pseudolabel_ini = torch.logical_not(pseudolabel_need_ini)
                    pl_ema = 0.01
                    pseudolabel_probs[pseudolabel_ini] = (1 - pl_ema) * pseudolabel_probs[
                        pseudolabel_ini
                    ] + pl_ema * new_pseudolabels[pseudolabel_ini]

                    # note pl initialised
                    pseudolabel_need_ini[batch_indices][mask] = 0

                    # Compute loss
                    cur_pseudolabels = pseudolabel_probs[batch_indices, :][mask]
                    loss = loss_f(logits[mask], cur_pseudolabels) * pseudo_frac
                elif args.label_smoothing in ("fw_soft", "fw_hard"):
                    new_pseudolabels = torch.zeros(
                        (all_y.shape[0], num_classes), device=device
                    )

                    # (num_pseudolabels, class)
                    pseudolabels_passed_threshold = torch.zeros(
                        (pseudo_y.shape[0], num_classes), device=device
                    )
                    pseudolabels_passed_threshold[
                        np.arange(pseudo_y.shape[0]), pseudo_y
                    ] = 1

                    # The index to assign pseudolabels_passed_threshold to
                    idx = torch.zeros_like(
                        pseudolabels_passed_threshold, dtype=torch.int64
                    )
                    idx[np.arange(pseudo_y.shape[0]), :] = (
                        batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                    )
                    new_pseudolabels.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )
                    pseudolabel_counts.scatter_add_(
                        dim=0, index=idx, src=pseudolabels_passed_threshold
                    )

                    # Initialize labels
                    ini_label_type = args.label_smoothing.split("_")[1]
                    if ini_label_type == "hard":
                        pseudolabel_probs[pseudolabel_need_ini] = new_pseudolabels[
                            pseudolabel_need_ini
                        ]
                    else:
                        # TODO: Implement soft initialization
                        pseudolabel_probs[pseudolabel_need_ini] = new_pseudolabels[
                            pseudolabel_need_ini
                        ]

                    # Get number of pseudolabel count to determine learning rate
                    num_pl_count = torch.sum(pseudolabel_counts.int(), axis=1)
                    cur_lr = pl_fw_lr[num_pl_count]  # (all)

                    pseudolabel_ini = torch.logical_not(pseudolabel_need_ini)
                    cur_lr = cur_lr[pseudolabel_ini][:, None]
                    pseudolabel_probs[pseudolabel_ini] = (1 - cur_lr) * pseudolabel_probs[
                        pseudolabel_ini
                    ] + cur_lr * new_pseudolabels[pseudolabel_ini]

                    # note pl initialised
                    pseudolabel_need_ini[batch_indices][mask] = 0

                    # Compute loss
                    cur_pseudolabels = pseudolabel_probs[batch_indices, :][mask]
                    loss = loss_f(logits[mask], cur_pseudolabels) * pseudo_frac
                else:
                    raise NotImplementedError()
            elif args.loss_type == "cl":
                # Repeat text_features [num_class] to [num_labels] based on pseudo_y
                full_text_features = text_features[pseudo_y]
                plabeled_image_features = image_features[mask]
                logits_per_images = 100.0 * plabeled_image_features @ full_text_features.T
                logits_per_text = 100.0 * full_text_features @ plabeled_image_features.T

                loss = adpu.get_contrastive_loss(
                    logits_per_images, logits_per_text, pseudo_y
                )
            else:
                raise NotImplementedError()

            if args.tent_lambda > 0.0:
                loss += args.tent_lambda * adpu.softmax_tent(logits)

            if args.iprior_tent_lambda > 0.0:
                loss += (
                    args.iprior_tent_lambda * adpu.iprior_softmax_entropy(logits).mean()
                )

            if args.pprior_tent_lambda > 0.0:
                if args.prior_estimate == "zeroshot-soft-update":
                    if total_step % 50 == 0:
                        with torch.no_grad():
                            gradless_logits = 100 * model.linear(features)
                        label_marginals = adpu.get_prior_estimate(
                            gradless_logits, hard=False
                        )

                if not args.marginal_smoothing:
                    entropy_loss = args.pprior_tent_lambda * adpu.pprior_softmax_entropy(
                        logits, prior=label_marginals, logger=logger
                    )
                else:
                    # Smooth prior by batch size ratio
                    # decay = 1 - args.batch_size / features.shape[0]
                    pmarginal_decay = 0.9
                    (
                        entropy_loss,
                        moving_population_marginal,
                    ) = adpu.pprior_softmax_entropy(
                        logits,
                        prior=label_marginals,
                        prev_marginal=moving_population_marginal,
                        decay=pmarginal_decay,
                        return_marginal=True,
                        logger=logger,
                    )
                    entropy_loss = entropy_loss * args.pprior_tent_lambda
                loss += entropy_loss
                logger.debug(f"Entropy loss: {entropy_loss}")
                save_kl_and_tv(label_marginals)

            if args.memo_lambda > 0.0:
                loss += args.memo_lambda * adpu.memo_softmax_entropy(logits)

            if args.param_smoothing == "avg":
                cur_param = model.linear.weight.detach().clone()
            loss.backward()
            optimizer.step()

            if args.param_smoothing:
                if args.param_smoothing == "avg":
                    updated_param = model.linear.weight.detach().clone()
                    model.linear.weight = torch.nn.Parameter(
                        (total_step + 1) / (total_step + 2) * cur_param
                        + 1 / (total_step + 2) * updated_param
                    )

            if args.display_steps and (step + 1) % args.display_steps == 0:
                if "visual" in args.param_to_optimize:
                    model.evaluate(
                        total_dataloader,
                        loss_f,
                        metrics=metrics,
                        metric_prefix="test",
                        logger=logger,
                    )
                else:
                    model.evaluate_features(
                        features,
                        all_y,
                        loss_f,
                        metrics=metrics,
                        metric_prefix="test",
                        logger=logger,
                    )

                with open(f"{folder}/results.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                if metrics["test_accuracy"][-1] < 1 / num_classes:
                    logger.info("Early stopping since test accuracy is worse than random")
                    return metrics

            step += 1
            total_step += 1

            if args.num_steps and total_step > args.num_steps:
                if "visual" in args.param_to_optimize:
                    model.evaluate(
                        total_dataloader,
                        loss_f,
                        metrics=metrics,
                        metric_prefix="test",
                        logger=logger,
                    )
                else:
                    model.evaluate_features(
                        features,
                        all_y,
                        loss_f,
                        metrics=metrics,
                        metric_prefix="test",
                        logger=logger,
                    )
                with open(f"{folder}/results.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                return metrics
        if args.display_epoch and (epoch + 1) % args.display_epoch == 0:
            if "visual" in args.param_to_optimize:
                model.evaluate(
                    total_dataloader,
                    loss_f,
                    metrics=metrics,
                    metric_prefix="test",
                    logger=logger,
                )
            else:
                model.evaluate_features(
                    features,
                    all_y,
                    loss_f,
                    metrics=metrics,
                    metric_prefix="test",
                    logger=logger,
                )

            with open(f"{folder}/results.json", "w") as f:
                json.dump(metrics, f, indent=2)
            if metrics["test_accuracy"][-1] < 1 / num_classes:
                logger.info("Early stopping since test accuracy is worse than random")
                return metrics
        if args.do_scheduling:
            scheduler.step()
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(1)

    args = get_args()
    if args.num_steps:
        args.num_steps = max(args.num_steps, args.num_epoch)

    tta_folder = "tta"
    folder = (
        f"{init_path.ROOT_path}/results/{tta_folder}/{args.dataname}"
        f"/model={args.clip_model}_pretrained={args.clip_pretrained}"
        f"/param_to_optimize={args.param_to_optimize}"
        f"/pseudolabel_src={args.pseudolabel_src}"
        f"/prior-estimate={args.prior_estimate}_loss_type={args.loss_type}"
        f"/ptent-lambda={args.pprior_tent_lambda}_tent-lambda={args.tent_lambda}"
        f"/label_smoothing={args.label_smoothing}_ts={args.do_teacher_student}"
        + (
            f"/scheduling={args.do_scheduling}_bias={args.linear_bias}_num_trained_prompts={args.num_trained_prompts}"
            if args.scheduler_type == "linear"
            else f"/scheduling={args.do_scheduling}-{args.scheduler_type}_bias={args.linear_bias}_num_trained_prompts={args.num_trained_prompts}"
        )
        + f"/threshold={args.confidence_threshold}_mm={args.momentum}"
        f"/bs={args.batch_size}_lr={args.lr}_wd={args.weight_decay}"
        f"/seed={args.seed}"
    )
    mscu.make_folder(folder)
    mscu.set_seed(args.seed)
    if mscu.is_completed(folder):
        print(f"Run existed in {folder}")
        exit()

    logger = mscu.get_logger(
        level=args.log_level, filename=f"{folder}/out.log", add_console=True
    )
    logger.info(f"Parameters: {json.dumps(vars(args), indent=2)}")

    metrics = main(args, folder, logger)
    logger.info(
        f"Final TTA Accuracy: {metrics['test_accuracy'][-1]}. Initial Acc: {metrics['Before TTA_accuracy'][0]}"
    )
    logger.info(f"Saved results to {folder}")

    # Plot Test Accuracy
    accs = metrics["test_accuracy"]
    b = list(np.arange(len(accs)) + 1)
    plt.plot(b, accs, label="Pseudolabel")

    if args.prior_estimate == "fewshot":
        valid_accs = metrics["validation_accuracy"]
        plt.plot(b, valid_accs, label="Labeled Accuracy")
    zeroshot_acc = mscu.get_zero_shot(
        args.dataname,
        parent_path=init_path.ROOT_path,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        logger=logger,
    )
    if zeroshot_acc is not None:
        plt.plot(
            [0],
            zeroshot_acc,
            marker="*",
            label="Zero-Shot CLIP",
            color="orange",
            linestyle="--",
        )
        plt.hlines(zeroshot_acc, xmin=0, xmax=b[-1], colors="orange", linestyles="dashed")
    plt.ylabel("Accuracy")
    plt.xlabel("Adaptation Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/acc.pdf")

    # Plot marginal
    if args.prior_estimate == "teacher-soft":
        plt.clf()
        tv = metrics["tv_dist"]
        b = list(np.arange(len(tv)) + 1)
        plt.plot(b, tv, label="TV distance")
        plt.ylabel("TV")
        plt.xlabel("Adaptation Steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}/tv.pdf")

        plt.clf()
        kl = metrics["kl_div"]
        b = list(np.arange(len(tv)) + 1)
        plt.plot(b, kl, label="TV distance")
        plt.ylabel("KL")
        plt.xlabel("Adaptation Steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}/kl.pdf")

    mscu.note_completed(folder)
