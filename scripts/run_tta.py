"""Perform test-time adpatation
"""
import argparse
import copy
import json
from collections import defaultdict

import init_path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils.adapt_utils as adpu
import utils.data_utils as datu
import utils.misc_utils as mscu
import utils.model_utils as modu
from configs.clip import AVAILABLE_CLIP_MODELS, AVAILABLE_CLIP_PRETRAINED_DATASETS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataname",
        type=str,
        default="cifar10",
        choices=datu.DATASET_GETTERS.keys(),
    )
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

    # Major ablation components
    parser.add_argument("--prior_estimate", type=str, default="zeroshot-soft")
    parser.add_argument("--klu_lambda", type=float, default=0.0)
    parser.add_argument("--klp_lambda", type=float, default=0.0)
    parser.add_argument("--tent_lambda", type=float, default=0.1)
    parser.add_argument("--pent_lambda", type=float, default=0.0)
    parser.add_argument("-c", "--confidence_threshold", type=float, default=0.9)
    parser.add_argument(
        "--label_smoothing",
        type=lambda input: mscu.type_or_none(input, str),
        default="ema",
    )
    parser.add_argument("--param_to_optimize", type=str, default="linear")
    parser.add_argument("--do_mean_teacher", type=mscu.boolean, default=False)

    # Minor ablation components
    parser.add_argument("--mean_teacher_coeff", type=float, default=0.999)
    parser.add_argument("--num_trained_prompts", type=int, default=1)
    parser.add_argument(
        "--label_smoothing_mu", type=float, default=0.1
    )  # default from https://arxiv.org/pdf/1512.00567.pdf
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
    parser.add_argument("--momentum", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=4242)

    parser.add_argument(
        "--display_steps", type=lambda input: mscu.type_or_none(input, int), default=1
    )
    parser.add_argument("--log_level", default="INFO", required=False)

    args = parser.parse_args()
    return args


def main(args, folder, logger):
    # Get model
    device = mscu.get_device()
    logger.info(f"Device: {device}")

    logger.info("Building Model")
    tokens = datu.get_tokens(args.dataname)
    if args.param_to_optimize == "textual":
        model = modu.CLIP_Text(
            tokens=tokens,
            device=device,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            num_trained_prompts=args.num_trained_prompts,
        )
    elif args.param_to_optimize == "linear":
        model = modu.CLIP_Linear(
            tokens=tokens,
            device=device,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            bias=False,
        )
    else:
        raise ValueError(f"Unrecognized param_to_optimize {args.param_to_optimize}")
    logger.info(f"Model built")

    if args.do_mean_teacher:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        logger.info(f"Doing Mean Teacher. Teacher Model built")


    # Get data
    saved_features = datu.get_saved_features(
        args.dataname, args.clip_model, args.clip_pretrained
    )
    features = saved_features["features"]
    all_y = saved_features["y"]
    num_samples = features.shape[0]
    logger.info(f"Number of data {num_samples} for {args.dataname}")

    if args.batch_size is None:  # None means full batch
        args.batch_size = num_samples

    if args.opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            # lr=args.lr * args.batch_size / num_samples,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=0.0,
        )
        logger.info("Use SGD optimizier")
    else:
        raise ValueError(f"Unrecognized optimizer {args.opt_type}")

    loss_f = torch.nn.CrossEntropyLoss()

    # Setup results dictionary
    metrics = defaultdict(list)
    metrics["tokens"] = tokens
    metrics.update(vars(args))

    # Setup dataset
    dataset = TensorDataset(features, all_y, torch.arange(features.shape[0]))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get label marginals
    num_classes = torch.max(all_y) + 1
    true_label_marginals = torch.from_numpy(
        mscu.get_label_marginals(all_y.cpu().numpy())
    )
    true_label_marginals = true_label_marginals.to(device)
    logger.info(f"True label marginals: {true_label_marginals}")

    if args.prior_estimate == "zeroshot-soft":
        with torch.no_grad():
            logits = 100 * model.linear(features)
        label_marginals_prior = logits.softmax(1).mean(0)
        logger.info("Use soft predictions for prior estimate")
    elif args.prior_estimate == "uniform":
        label_marginals_prior = torch.ones(num_classes, device=device) / num_classes
        logger.info("Use uniform distribution for prior estimate")
    elif args.prior_estimate == "true":
        label_marginals_prior = true_label_marginals
        logger.info("Use true label distribution for prior estimate")
    else:
        raise NotImplementedError()

    model.evaluate_features(
        features,
        all_y,
        loss_f,
        metrics=metrics,
        metric_prefix="Before TTA",
        logger=logger,
    )

    if args.label_smoothing is not None:
        pseudolabel_probs = torch.zeros((all_y.shape[0], num_classes), device=device)
        logger.info(f"Smoothing label with {args.label_smoothing}")

    step = 0

    for epoch in range(args.num_epoch):
        for data in tqdm(dataloader, desc=f"Epoch {epoch}"):
            X, _, batch_indices = data
            batch_indices = batch_indices.to(device)
            X = X.to(device)

            # GD
            optimizer.zero_grad()
            if args.param_to_optimize == "textual":
                logits = model.forward_image_features(X, train=True)
            elif args.param_to_optimize == "linear":
                logits = 100.0 * model.linear(X)
            else:
                raise ValueError(
                    f"Unrecognized param_to_optimize {args.param_to_optimize}"
                )

            # Assign the logits used for pseudolabeling
            if args.do_mean_teacher:
                adpu.update_teacher_with_student(teacher_model, model, decay=args.mean_teacher_coeff)
                if step == 0:
                    logger.info(
                        f"Doing mean teacher with decay {args.mean_teacher_coeff}"
                    )

                with torch.no_grad():
                    if args.param_to_optimize == "textual":
                        teacher_logits = teacher_model.forward_image_features(
                            X, train=False
                        )
                    elif args.param_to_optimize == "linear":
                        teacher_logits = 100.0 * teacher_model.linear(X)
                    else:
                        raise NotImplementedError()
                pseudolabeling_logits = teacher_logits
            else:
                pseudolabeling_logits = logits

            _, pseudo_y, pseudo_frac, mask = adpu.get_pseudo_labels(
                pseudolabeling_logits,
                confidence_threshold=args.confidence_threshold,
                return_mask=True,
            )
            metrics["pseudo_fracs"].append(pseudo_frac.detach().cpu().item())

            # Do label smoothing
            if args.label_smoothing is None:
                loss = loss_f(logits[mask], pseudo_y) * pseudo_frac
            elif args.label_smoothing == "usmooth":
                if step == 0:
                    logger.info(
                        f"Smoothing labels with uniform by {args.label_smoothing_mu}"
                    )
                pseudolabels = torch.zeros(
                    (pseudo_y.shape[0], num_classes), device=device
                )
                pseudolabels[np.arange(pseudo_y.shape[0]), pseudo_y] = 1.0
                uniform_label = torch.ones(num_classes, device=device) / num_classes
                pseudolabels = (
                    pseudolabels * (1 - args.label_smoothing_mu)
                    + uniform_label * args.label_smoothing_mu
                )
                loss = loss_f(logits[mask], pseudolabels) * pseudo_frac
            elif args.label_smoothing == "ema":
                if args.confidence_threshold > 0.0:
                    logger.warning("EMA does not currently support thresholding pseudolabels")
                    quit()
                pl_ema = 0.4 # parameter chosen in https://arxiv.org/pdf/1610.02242.pdf
                if step == 0:
                    logger.info(
                        f"Smoothing labels with exponential moving averages by {pl_ema}"
                    )
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
                idx = torch.zeros_like(pseudolabels_passed_threshold, dtype=torch.int64)
                idx[np.arange(pseudo_y.shape[0]), :] = (
                    batch_indices[torch.where(mask)[0]].repeat(num_classes, 1).T
                )
                new_pseudolabels.scatter_add_(
                    dim=0, index=idx, src=pseudolabels_passed_threshold
                )

                pseudolabel_probs = (
                    1 - pl_ema
                ) * pseudolabel_probs + pl_ema * new_pseudolabels

                # Compute loss
                if step + 1 >= 80:
                    ema_weight = 1
                else:
                    ema_weight = np.exp(-5 * (1 - (step+1) / 80)**2)

                bias_correction = (1 - (1 - pl_ema) ** (step+1))
                cur_pseudolabels = pseudolabel_probs[batch_indices, :][mask] / bias_correction
                logger.info(f"Step {step} with bias correction {bias_correction}")
                loss = ema_weight * loss_f(logits[mask], cur_pseudolabels) * pseudo_frac

            # Add test-time entropy
            if args.tent_lambda > 0.0:
                loss += args.tent_lambda * adpu.softmax_tent(logits)
                if step == 0:
                    logger.info("Add test-time entropy to loss")

            # Add population entropy
            if args.pent_lambda > 0.0:
                entropy_loss = args.pent_lambda * adpu.pprior_softmax_entropy(
                    logits, prior=label_marginals_prior, logger=logger
                )
                loss += entropy_loss
                if step == 0:
                    logger.info("Add population cross entropy -q\log p to loss")

            # Add KL uniform entropy
            if args.klu_lambda > 0.0:
                loss += args.klu_lambda * adpu.kl_uniform(logits)
                if step == 0:
                    logger.info("Add KL(p||uniform) := p\log p to loss")

            # Add KL prior entropy
            if args.klp_lambda > 0.0:
                loss += args.klp_lambda * adpu.kl_prior(
                    logits, prior=label_marginals_prior
                )
                if step == 0:
                    logger.info("Add KL(p||q) := p (\log p - \log q) to loss")

            loss.backward()
            optimizer.step()

            # Display
            if args.display_steps and (step + 1) % args.display_steps == 0:
                model.evaluate_features(
                    features,
                    all_y,
                    loss_f,
                    metrics=metrics,
                    metric_prefix="test",
                    logger=logger,
                )

            step += 1

            # Terminate
            if args.num_steps and step > args.num_steps:
                model.evaluate_features(
                    features,
                    all_y,
                    loss_f,
                    metrics=metrics,
                    metric_prefix="test",
                    logger=logger,
                )
                logger.info(
                    f"Terminating b/c number of steps exceed maximum number {args.num_steps}"
                )
                with open(f"{folder}/results.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                return metrics
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(1)

    args = get_args()
    if args.num_steps:
        args.num_steps = max(args.num_steps, args.num_epoch)

    tta_folder = "tta"
    param_text = (
        "" if args.param_to_optimize == "linear" else f"_param={args.param_to_optimize}"
    )
    epoch_text = "" if args.num_epoch == 200 else f"_epoch={args.num_epoch}"
    mt_text = "" if not args.do_mean_teacher else f"_mt={args.do_mean_teacher}"

    folder = (
        f"{init_path.ROOT_path}/results/{tta_folder}/{args.dataname}"
        f"/model={args.clip_model}_pretrained={args.clip_pretrained}"
        + param_text
        + f"/prior-estimate={args.prior_estimate}_ls={args.label_smoothing}"
        + mt_text
        + f"/klu-lambda={args.klu_lambda}_klp-lambda={args.klp_lambda}_tent={args.tent_lambda}_pent={args.pent_lambda}"
        f"/threshold={args.confidence_threshold}_mm={args.momentum}"
        f"/bs={args.batch_size}_lr={args.lr}" + epoch_text + f"/seed={args.seed}"
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
    with open(f"{folder}/results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved results to {folder}")

    # Plot Test Accuracy
    accs = metrics["test_accuracy"]
    b = list(np.arange(len(accs)) + 1)
    plt.plot(b, accs, label="Pseudolabel")

    # Plot zeroshot accuracy
    zeroshot_acc = metrics["Before TTA_accuracy"][0]
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
