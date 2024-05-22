import numpy as np
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.misc_utils as mscu

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"

LOGGER = mscu.get_logger()


class CLIP_Full(torch.nn.Module):
    def __init__(
        self,
        tokens,
        device,
        clip_model=DEFAULT_CLIP_MODEL,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        num_trained_prompts=1,
    ):
        super().__init__()
        self.tokens = tokens
        self.device = device
        self.featurizer, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.num_trained_prompts = num_trained_prompts

        # Build text features
        with torch.no_grad():
            self.initial_text_features = self.featurize_tokens(tokens)

        # Initialize linear layer
        self.linear = torch.nn.Linear(
            self.initial_text_features.shape[1],
            self.initial_text_features.shape[0],
            bias=False,
        ).to(device)
        self.linear.weight = torch.nn.Parameter(
            self.initial_text_features.detach().clone().float()
        )

        LOGGER.info(
            f"CLIP model initialized. Model {clip_model} pretrained {clip_pretrained}"
        )

    def featurize_tokens(self, tokens, train=False):
        text_features = []
        for cls_tokens in tokens:
            if train:  # Limit number of prompts trained for memory
                cls_tokens = np.random.choice(
                    cls_tokens,
                    size=min([self.num_trained_prompts, len(cls_tokens)]),
                    replace=False,
                ).tolist()
            text = self.tokenizer(cls_tokens).to(self.device)
            cur_text_features = self.featurizer.encode_text(text)
            cur_text_features = cur_text_features / cur_text_features.norm(
                dim=-1, keepdim=True
            )
            cur_text_features = torch.mean(cur_text_features, dim=0)
            cur_text_features = cur_text_features / cur_text_features.norm()
            text_features.append(cur_text_features)
        return torch.vstack(text_features)

    def get_features(self, images, train=False):
        image_features = self.featurizer.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.featurize_tokens(self.tokens, train=train)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    def forward(self, images, train=False):
        image_features, text_features = self.get_features(images, train=train)
        return 100.0 * image_features @ text_features.T

    def forward_image_features(self, image_features, train=False):
        text_features = self.featurize_tokens(self.tokens, train=train)
        logits = 100.0 * image_features @ text_features.T
        return logits

    def evaluate(
        self, dataloader, loss_f, metrics=None, metric_prefix="test", logger=None
    ):
        self.eval()
        with torch.no_grad():
            all_y = []
            all_ypred = []
            total_loss = 0

            for X, y in tqdm(dataloader, desc="Evaluating model", leave=False):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.forward(X)
                batch_loss = loss_f(logits, y)
                total_loss += batch_loss * y.shape[0]
                ypred = logits.argmax(-1)

                all_y.append(y)
                all_ypred.append(ypred)
            all_y, all_ypred = torch.cat(all_y), torch.cat(all_ypred)

        avg_loss = total_loss / len(dataloader.dataset)
        avg_loss = avg_loss.cpu().numpy().item()
        acc = torch.sum(all_y == all_ypred) / all_y.numel()
        acc = acc.cpu()

        if logger:
            logger.info(f"loss: {avg_loss:.7f}" f" acc: {acc:03.3f}")

        metrics[f"{metric_prefix}_loss"].append(avg_loss)
        metrics[f"{metric_prefix}_accuracy"].append(acc.numpy().item())
        self.train()

    def evaluate_features(
        self, features, y, loss_f, metrics=None, metric_prefix="test", logger=None
    ):
        with torch.no_grad():
            text_features = self.featurize_tokens(self.tokens, train=False)
            logits = 100.0 * features @ text_features.T
        avg_loss = loss_f(logits, y)
        avg_loss = avg_loss.cpu().numpy().item()

        ypred = logits.argmax(-1)
        acc = torch.sum(y == ypred) / y.numel()
        acc = acc.cpu()

        if logger is not None:
            logger.debug(f"logits: {logits}")
            logger.debug(f"ypred: {ypred}")
            logger.debug(
                f"ypred marginals: {mscu.get_label_marginals(ypred.cpu().numpy())}"
            )
            logger.info(
                f"{metric_prefix} loss: {avg_loss:.7f}"
                f" {metric_prefix} acc: {acc:03.3f}"
            )

        if metrics is not None:
            metrics[f"{metric_prefix}_loss"].append(avg_loss)
            metrics[f"{metric_prefix}_accuracy"].append(acc.numpy().item())


class CLIP_Linear(torch.nn.Module):
    def __init__(
        self,
        tokens,
        device,
        clip_model=DEFAULT_CLIP_MODEL,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        bias=False,
    ):
        """Initiate CLIP models w/ tunable linear layer initiated with text featues
            tokens (List[List[str]]): a (num_tokens, num_templates) sized tokens
        """
        super().__init__()
        self.tokens = tokens
        self.featurizer, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )

        # Build text features
        tokenizer = open_clip.get_tokenizer(clip_model)
        self.text_features = []
        with torch.no_grad():
            for cls_tokens in tokens:
                text = tokenizer(cls_tokens).to(device)
                cur_text_features = self.featurizer.encode_text(text)
                cur_text_features /= cur_text_features.norm(dim=-1, keepdim=True)
                cur_text_features = torch.mean(cur_text_features, dim=0)
                cur_text_features /= cur_text_features.norm()
                self.text_features.append(cur_text_features)
        self.text_features = torch.vstack(self.text_features)

        # Initialize linear layer
        self.linear = torch.nn.Linear(
            self.text_features.shape[1], self.text_features.shape[0], bias=bias
        ).to(device)
        if bias:
            torch.nn.init.zeros_(self.linear.bias)
        self.linear.weight = torch.nn.Parameter(
            self.text_features.detach().clone().float()
        )

        if clip_model == "ViT-B-32":
            self.feature_dim = 512
        elif clip_model == "ViT-B-16-plus-240":
            self.feature_dim = 640
        elif clip_model == "ViT-L-14-336":
            self.feature_dim = 768
        elif clip_model == "RN50":
            self.feature_dim = 1024
        self.device = device

        LOGGER.info(
            f"CLIP model initialized. Model {clip_model} pretrained {clip_pretrained}"
        )

    def featurize(self, images):
        image_features = self.featurizer.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, images):
        with torch.no_grad():
            image_features = self.featurizer.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100 * self.linear(image_features.float())
        return logits

    def evaluate(
        self, dataloader, loss_f, metrics=None, metric_prefix="test", logger=None
    ):
        self.eval()
        with torch.no_grad():
            all_y = []
            all_ypred = []
            total_loss = 0

            for X, y in tqdm(dataloader, desc="Evaluating model", leave=False):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.forward(X)
                batch_loss = loss_f(logits, y)
                total_loss += batch_loss * y.shape[0]
                ypred = logits.argmax(-1)

                all_y.append(y)
                all_ypred.append(ypred)
            all_y, all_ypred = torch.cat(all_y), torch.cat(all_ypred)

        avg_loss = total_loss / len(dataloader.dataset)
        avg_loss = avg_loss.cpu().numpy().item()
        acc = torch.sum(all_y == all_ypred) / all_y.numel()
        acc = acc.cpu()

        if logger:
            logger.info(f"loss: {avg_loss:.7f}" f" acc: {acc:03.3f}")

        metrics[f"{metric_prefix}_loss"].append(avg_loss)
        metrics[f"{metric_prefix}_accuracy"].append(acc.numpy().item())
        self.train()

    def evaluate_features(
        self, features, y, loss_f, metrics=None, metric_prefix="test", logger=None
    ):
        with torch.no_grad():
            logits = self.linear(features)
        avg_loss = loss_f(logits, y)
        avg_loss = avg_loss.cpu().numpy().item()

        ypred = logits.argmax(-1)
        acc = torch.sum(y == ypred) / y.numel()
        acc = acc.cpu()

        if logger is not None:
            logger.debug(f"logits: {logits}")
            logger.debug(f"ypred: {ypred}")
            logger.debug(
                f"ypred marginals: {mscu.get_label_marginals(ypred.cpu().numpy())}"
            )
            logger.info(
                f"{metric_prefix} loss: {avg_loss:.7f}"
                f" {metric_prefix} acc: {acc:03.3f}"
            )

        if metrics is not None:
            metrics[f"{metric_prefix}_loss"].append(avg_loss)
            metrics[f"{metric_prefix}_accuracy"].append(acc.numpy().item())


class CLIP_Text(torch.nn.Module):
    def __init__(
        self,
        tokens,
        device,
        clip_model=DEFAULT_CLIP_MODEL,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        num_trained_prompts=1,
    ):
        """Initiate CLIP models w/ tunable linear layer initiated with text featues
            tokens (List[List[str]]): a (num_tokens, num_templates) sized tokens
        """
        super().__init__()
        self.device = device
        self.tokens = tokens
        self.num_trained_prompts = num_trained_prompts
        self.featurizer, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )

        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Build text features
        with torch.no_grad():
            self.initial_text_features = self.featurize_tokens(tokens)

        # Initialize linear layer
        self.linear = torch.nn.Linear(
            self.initial_text_features.shape[1],
            self.initial_text_features.shape[0],
            bias=False,
        ).to(device)
        self.linear.weight = torch.nn.Parameter(
            self.initial_text_features.detach().clone().float()
        )

        if clip_model == "ViT-B-32":
            self.feature_dim = 512
        elif clip_model == "ViT-B-16-plus-240":
            self.feature_dim = 640
        elif clip_model == "ViT-L-14-336":
            self.feature_dim = 768
        elif clip_model == "RN50":
            self.feature_dim = 1024

        LOGGER.info(
            f"CLIP model initialized. Model {clip_model} pretrained {clip_pretrained}"
        )

    def featurize_images(self, images):
        image_features = self.featurizer.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def featurize_tokens(self, tokens, train=False):
        text_features = []
        for cls_tokens in tokens:
            if train:  # Limit number of prompts trained for memory
                cls_tokens = np.random.choice(
                    cls_tokens,
                    size=min([self.num_trained_prompts, len(cls_tokens)]),
                    replace=False,
                ).tolist()
            text = self.tokenizer(cls_tokens).to(self.device)
            cur_text_features = self.featurizer.encode_text(text)
            cur_text_features = cur_text_features / cur_text_features.norm(
                dim=-1, keepdim=True
            )
            cur_text_features = torch.mean(cur_text_features, dim=0)
            cur_text_features = cur_text_features / cur_text_features.norm()
            text_features.append(cur_text_features)
        return torch.vstack(text_features)

    def forward_image_features(self, image_features, train=False):
        text_features = self.featurize_tokens(self.tokens, train=train)
        logits = 100.0 * image_features @ text_features.T
        return logits

    def evaluate_features(
        self, image_features, y, loss_f, metrics=None, metric_prefix="test", logger=None
    ):
        self.eval()
        with torch.no_grad():
            logits = self.forward_image_features(image_features)
        avg_loss = loss_f(logits, y)
        avg_loss = avg_loss.cpu().numpy().item()

        ypred = logits.argmax(-1)
        acc = torch.sum(y == ypred) / y.numel()
        acc = acc.cpu()

        if logger is not None:
            logger.debug(f"logits: {logits}")
            logger.debug(f"ypred: {ypred}")
            logger.debug(
                f"ypred marginals: {mscu.get_label_marginals(ypred.cpu().numpy())}"
            )
            logger.info(
                f"{metric_prefix} loss: {avg_loss:.7f}"
                f" {metric_prefix} acc: {acc:03.3f}"
            )

        if metrics is not None:
            metrics[f"{metric_prefix}_loss"].append(avg_loss)
            metrics[f"{metric_prefix}_accuracy"].append(acc.numpy().item())

        self.train()


class TeacherTeST(torch.nn.Module):
    def __init__(self, tokens, featurizer, feature_dim, device) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.feature_dim = feature_dim
        self.device = device

        # Linear layer for predicting between strong and weakly augmented images
        self.linear_predictor = torch.nn.Linear(self.feature_dim, self.feature_dim).to(
            device
        )
        torch.nn.init.zeros_(self.linear_predictor.bias)
        self.linear_predictor.weight.data.copy_(torch.eye(self.feature_dim))

        # Linear layer representing text features
        self.tokens = tokens
        self.text = clip.tokenize(tokens).to(device)
        with torch.no_grad():
            # (10, 512)
            self.text_features = self.featurizer.encode_text(self.text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        # Initialize linear layer
        self.linear = torch.nn.Linear(
            self.text_features.shape[1], self.text_features.shape[0]
        ).to(device)
        torch.nn.init.zeros_(self.linear.bias)
        self.linear.weight = torch.nn.Parameter(
            self.text_features.detach().clone().float()
        )

    def get_feature_distance(self, weak_images, strong_images):
        weak_feat = self.featurizer.encode_image(weak_images)
        strong_feat = self.featurizer.encode_image(strong_images)
        strong_feat = self.linear_predictor(strong_feat.float())

        import pdb

        pdb.set_trace()
        return torch.norm(weak_feat - strong_feat) ** 2 / weak_images.shape[0]

    def forward(self, images):
        with torch.no_grad():
            image_features = self.featurizer.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100 * self.linear(image_features.float())
        return logits


def initialize_torchvision_model(name, d_out, pretrained=True):
    import antialiased_cnns

    # get constructor and last layer names
    if name == "wideresnet50":
        constructor_name = "wide_resnet50_2"
        last_layer_name = "fc"
    elif name == "densenet121":
        constructor_name = name
        last_layer_name = "classifier"
    elif name in ("resnet18", "resnet34", "resnet50", "resnet101"):
        constructor_name = name
        last_layer_name = "fc"
    elif name in ("efficientnet_b0"):
        constructor_name = name
        last_layer_name = "classifier"
    else:
        raise ValueError(f"Torchvision model {name} not recognized")
    # construct the default model, which has the default last layer
    constructor = getattr(antialiased_cnns, constructor_name)
    model = constructor(pretrained=pretrained)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = nn.Identity(d_features)
        model.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    model.feature_dim = d_features

    return model
