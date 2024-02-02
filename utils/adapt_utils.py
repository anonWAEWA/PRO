from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.misc_utils as mscu


def get_pseudo_labels(logits, confidence_threshold=0.8, true_y=None, return_mask=False):
    """
    Input:
        logits (Tensor): Multi-class logits of size (batch_size, ..., n_classes).
        confidence_threshold (float): In [0,1]
    Output:
        pseudo_logits (Tensor): Filtered version of logits, discarding any rows (examples) that
                                   have no predictions with confidence above confidence_threshold.
        pseudo_y (Tensor): Corresponding hard-pseudo-labeled version of logits. All
                                     examples with confidence below confidence_threshold are discarded.
        pseudo_frac (float): Fraction of examples not discarded.
    """
    mask = torch.max(F.softmax(logits, -1), -1)[0] >= confidence_threshold
    pseudo_y = logits.argmax(-1)
    pseudo_y = pseudo_y[mask]
    if true_y is not None:
        n_same = torch.sum(pseudo_y == true_y[mask])
        print(f"Num labels: {len(pseudo_y)}. Perc correct: {n_same / len(pseudo_y)}")
    pseudo_logits = logits[mask]
    pseudo_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()

    if not return_mask:
        return pseudo_logits, pseudo_y, pseudo_frac
    else:
        return pseudo_logits, pseudo_y, pseudo_frac, mask


def get_pseudo_labels_from_logprob(
    logprob, confidence_threshold=0.8, true_y=None, return_mask=False
):
    """
    Input:
        logprob (Tensor): Multi-class log probabilities of size (batch_size, ..., n_classes).
        confidence_threshold (float): In [0,1]
    Output:
        pseudo_logprobs (Tensor): Filtered version of logits, discarding any rows (examples) that
                                   have no predictions with confidence above confidence_threshold.
        pseudo_y (Tensor): Corresponding hard-pseudo-labeled version of logits. All
                                     examples with confidence below confidence_threshold are discarded.
        pseudo_frac (float): Fraction of examples not discarded.
    """
    mask = torch.max(torch.exp(logprob), -1)[0] >= confidence_threshold
    pseudo_y = logprob.argmax(-1)
    pseudo_y = pseudo_y[mask]
    if true_y is not None:
        n_same = torch.sum(pseudo_y == true_y[mask])
        print(f"Num labels: {len(pseudo_y)}. Perc correct: {n_same / len(pseudo_y)}")
    pseudo_logprobs = logprob[mask]
    pseudo_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()

    if not return_mask:
        return pseudo_logprobs, pseudo_y, pseudo_frac
    else:
        return pseudo_logprobs, pseudo_y, pseudo_frac, mask


def iprior_softmax_entropy(x: torch.Tensor, prior=None) -> torch.Tensor:
    """Entropy of softmax distribution from logits.
    Computed with with prior on label marginal of individual samples.
    """
    # If prior is not provided, assume unifrom prior
    if prior is None:
        prior = torch.ones(x.shape[1], device=x.device) / x.shape[1]

    return -(x.log_softmax(1) * prior[None, :]).sum(1)


def pprior_softmax_entropy(
    x: torch.Tensor, prior=None, prev_marginal=None, decay=None, return_marginal=False, logger=None
) -> torch.Tensor:
    """Entropy of softmax distribution from logits.
    Computed with with prior on label marginal of population.
    """
    # If prior is not provided, assume unifrom prior
    if prior is None:
        prior = torch.ones(x.shape[1], device=x.device) / x.shape[1]

    p = x.softmax(1)
    population_marginal = p.mean(dim=0)
    if prev_marginal is not None and decay is not None:
        population_marginal = decay * prev_marginal + (1 - decay) * population_marginal
    if logger is not None:
        logger.debug(f"Predicted population {population_marginal}")

    if not return_marginal:
        return -(prior * torch.log(population_marginal)).sum()
    else:
        return -(prior * torch.log(population_marginal)).sum(), population_marginal.detach()


def kl_uniform(logits):
    population_marginal = logits.softmax(1).mean(dim=0)
    return (population_marginal * torch.log(population_marginal)).sum()

def kl_prior(logits, prior):
    population_marginal = logits.softmax(1).mean(dim=0)
    return (population_marginal * (torch.log(population_marginal) - torch.log(prior))).sum()

def softmax_tent(x: torch.Tensor) -> torch.Tensor:
    """Entropy term to minimize as described in https://github.com/DequanWang/tent
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)


def memo_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.
    As described in https://arxiv.org/pdf/2110.09506.pdf.
    """
    p = x.softmax(1)
    population_marginal = p.mean(dim=0)

    return -(population_marginal * torch.log(population_marginal)).sum()


def get_prior_estimate(logits, hard=True, threshold=None, percentage=None):
    probs = logits.softmax(1)

    if threshold is not None:
        mask = torch.max(probs, -1)[0] >= threshold
        probs = probs[mask]
    elif percentage is not None:
        _, mask = torch.topk(torch.max(probs, -1)[0], k=int(percentage * logits.shape[0]))
        probs = probs[mask]

    if hard:
        ypred = probs.argmax(-1)
        label_marginals = torch.from_numpy(mscu.get_label_marginals(ypred.cpu().numpy()))
        return label_marginals.to(logits.device)
    else:
        return probs.mean(0)


def split_random_idc(original_idc, num_split, seed):
    np.random.seed(seed)

    split_idc = np.random.choice(original_idc, num_split, replace=False)
    rest_idc = np.setdiff1d(original_idc, split_idc, assume_unique=True)
    return split_idc, rest_idc


def get_fixmatch_loss(
    model, X_weak, X_strong, confidence_threshold=0.9, criterion=nn.CrossEntropyLoss()
):
    # Filter training sample by threshold
    model.eval()
    with torch.no_grad():
        weak_logits = model(X_weak)
        mask = torch.max(F.softmax(weak_logits, -1), -1)[0] >= confidence_threshold
        unlabeled_pseudo_y = weak_logits.argmax(-1)
        unlabeled_pseudo_y = unlabeled_pseudo_y[mask]
    model.train()

    # Filter strongly augmented and train
    strong_logits = model(X_strong[[mask]])
    pseudolabels_kept_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()

    fixmatch_loss = pseudolabels_kept_frac * criterion(strong_logits, unlabeled_pseudo_y)

    return fixmatch_loss


def get_contrastive_loss(logits_per_image, logits_per_text, ground_labels):
    """Taken from FLYP https://github.com/locuslab/FLYP/blob/main/clip/loss.py"""
    ground_labels_repeated = ground_labels.view(1, -1).repeat(
        logits_per_image.shape[0], 1
    )
    equal_labels = (ground_labels_repeated == ground_labels.view(-1, 1)).type(torch.float)
    labels = equal_labels / torch.sum(equal_labels, dim=1).view(-1, 1)

    total_loss = (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2
    return total_loss


def update_teacher_with_student(teacher, student, decay):
    """Code snippet taken from https://www.zijianhu.com/post/pytorch/ema/"""
    model_params = OrderedDict(student.named_parameters())
    teacher_param = OrderedDict(teacher.named_parameters())

    # check if both model contains the same set of keys
    assert model_params.keys() == teacher_param.keys()

    with torch.no_grad():
        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            teacher_param[name].sub_((1.0 - decay) * (teacher_param[name] - param))

    model_buffers = OrderedDict(student.named_buffers())
    shadow_buffers = OrderedDict(teacher.named_buffers())

    # check if both model contains the same set of keys
    assert model_buffers.keys() == shadow_buffers.keys()

    for name, buffer in model_buffers.items():
        # buffers are copied
        shadow_buffers[name].copy_(buffer)
