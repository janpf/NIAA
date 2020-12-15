from argparse import ArgumentParser
import logging
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn, optim
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

from train_pre.dataset import SSPexelsSmall as SSPexels
from train_pre.losses import EfficientRankingLoss, h
from train_pre.preprocess_images import mapping


class CheckpointModule(nn.Module):
    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        return checkpoint_sequential(self.module, self.num_segments, *inputs)


class IA(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--margin", type=float, default=0.2)
        parser.add_argument("--lr_decay_rate", type=float, default=0.5)
        parser.add_argument("--lr_patience", type=float, default=3)
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=40)

        parser.add_argument("--scores", type=str, default=None)  # "one", "three", None; None is a valid input
        parser.add_argument("--change_regress", action="store_true")
        parser.add_argument("--change_class", action="store_true")
        return parser

    def __init__(self, scores: str, change_regress: bool, change_class: bool, mapping, margin, pretrained: bool = False, fix_features: bool = False):
        super().__init__()
        self.scores = scores
        self.change_regress = change_regress
        self.change_class = change_class

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = torchvision.models.mobilenet.mobilenet_v2(pretrained=pretrained)
        self.feature_count = 1280

        if fix_features:
            for param in base_model.features.parameters():
                param.requires_grad = False

        self.mapping = mapping
        self.margin = margin
        self.features = CheckpointModule(module=base_model.features, num_segments=len(base_model.features))

        self.mseloss = nn.MSELoss()
        self.celoss = nn.CrossEntropyLoss()
        self.T = 50
        self.save_hyperparameters()

        # a single score giving the aesthetics
        if self.scores == "one":
            # fmt: off
            self.score = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=1),
                nn.Sigmoid())
            # fmt: on
            nn.init.xavier_uniform(self.score[1].weight)

        # three scores giving three separate aesthetics
        elif self.scores == "three":
            # fmt: off
            self.styles_score = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=1),
                nn.Sigmoid())
            self.technical_score = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=1),
                nn.Sigmoid())
            self.composition_score = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=1),
                nn.Sigmoid())
            # fmt: on

            nn.init.xavier_uniform(self.styles_score[1].weight)
            nn.init.xavier_uniform(self.technical_score[1].weight)
            nn.init.xavier_uniform(self.composition_score[1].weight)

        # regression predicting the level of a distortion
        if self.change_regress:
            # fmt: off
            self.styles_change_strength = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["styles"])))
            self.technical_change_strength = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["technical"])))
            self.composition_change_strength = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["composition"])))
            # fmt: on
            nn.init.xavier_uniform(self.styles_change_strength[1].weight)
            nn.init.xavier_uniform(self.technical_change_strength[1].weight)
            nn.init.xavier_uniform(self.composition_change_strength[1].weight)

        # predicting the class of a distortion
        if self.change_class:
            # fmt: off
            self.styles_change_class = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["styles"])))
            self.technical_change_class = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["technical"])))
            self.composition_change_class = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["composition"])))
            # fmt: on
            nn.init.xavier_uniform(self.styles_change_class[1].weight)
            nn.init.xavier_uniform(self.technical_change_class[1].weight)
            nn.init.xavier_uniform(self.composition_change_class[1].weight)

    def forward(self, x: torch.Tensor):
        x = self.features(x)

        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        result_dict = dict()

        if self.scores == "one":
            result_dict["score"] = self.score(x)

        elif self.scores == "three":
            result_dict["styles_score"] = self.styles_score(x)
            result_dict["technical_score"] = self.technical_score(x)
            result_dict["composition_score"] = self.composition_score(x)

        if self.change_regress:
            result_dict["styles_change_strength"] = self.styles_change_strength(x)
            result_dict["technical_change_strength"] = self.technical_change_strength(x)
            result_dict["composition_change_strength"] = self.composition_change_strength(x)

        if self.change_class:
            result_dict["styles_change_class"] = self.styles_change_class(x)
            result_dict["technical_change_class"] = self.technical_change_class(x)
            result_dict["composition_change_class"] = self.composition_change_class(x)

        return result_dict

    def _calc_loss(self, batch):

        ranking_losses_step: Dict[str, List] = dict()
        change_regress_losses_step: Dict[str, List] = dict()
        change_class_losses_step: Dict[str, List] = dict()

        for distortion in ["styles", "technical", "composition"]:
            ranking_losses_step[distortion] = []
            change_regress_losses_step[distortion] = []
            change_class_losses_step[distortion] = []

        original: Dict[str, torch.Tensor] = self(batch["original"].to(self.device))
        crop_orig: Dict[str, torch.Tensor] = self(batch["crop_original"].to(self.device))
        rotate_orig: Dict[str, torch.Tensor] = self(batch["rotate_original"].to(self.device))

        for distortion in ["styles", "technical", "composition"]:
            erloss = EfficientRankingLoss(margin=self.margin[distortion])
            for parameter in self.mapping[distortion]:
                for polarity in self.mapping[distortion][parameter]:
                    results = dict()
                    for change in self.mapping[distortion][parameter][polarity]:
                        results[change] = self(batch[change].to(self.device))
                    logging.debug(polarity)
                    if self.scores == "one":
                        if "crop" in parameter:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\tcrop")
                            ranking_losses_step[distortion].append(erloss(crop_orig, x=results, polarity=polarity, score="score"))
                        elif "rotate" in parameter:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\trotate")
                            ranking_losses_step[distortion].append(erloss(rotate_orig, x=results, polarity=polarity, score="score"))
                        else:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\torig")
                            ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score="score"))
                    elif self.scores == "three":
                        if "crop" in parameter:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\tcrop\t{distortion}_score")
                            ranking_losses_step[distortion].append(erloss(crop_orig, x=results, polarity=polarity, score=f"{distortion}_score"))
                        elif "rotate" in parameter:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\trotate\t{distortion}_score")
                            ranking_losses_step[distortion].append(erloss(rotate_orig, x=results, polarity=polarity, score=f"{distortion}_score"))
                        else:
                            logging.debug(f"{distortion}\t{parameter}\t{polarity}\tranking\torig\t{distortion}_score")
                            ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score"))

                    if self.change_regress:
                        if polarity == "pos":  # originals get regressed only once
                            if "crop" in parameter:
                                logging.debug(f"{distortion}\t{parameter}\t{polarity}\tregress\tcrop\t{distortion}_score")
                                to_be_regressed_param_column = crop_orig[f"{distortion}_change_strength"][:, list(self.mapping[distortion].keys()).index(parameter)]
                            elif "rotate" in parameter:
                                logging.debug(f"{distortion}\t{parameter}\t{polarity}\tregress\trotate\t{distortion}_score")
                                to_be_regressed_param_column = rotate_orig[f"{distortion}_change_strength"][:, list(self.mapping[distortion].keys()).index(parameter)]
                            else:
                                logging.debug(f"{distortion}\t{parameter}\t{polarity}\tregress\torig\t{distortion}_score")
                                to_be_regressed_param_column = original[f"{distortion}_change_strength"][:, list(self.mapping[distortion].keys()).index(parameter)]
                            correct_list = [0] * len(to_be_regressed_param_column)
                            correct_list = torch.Tensor(correct_list)

                            change_regress_losses_step[distortion].append(self.mseloss(to_be_regressed_param_column, correct_list.to(self.device)))

                        for change in self.mapping[distortion][parameter][polarity]:
                            if polarity == "pos":
                                correct_value = self.mapping["change_steps"][distortion][parameter][polarity] * (self.mapping[distortion][parameter][polarity].index(change) + 1)
                            if polarity == "neg":
                                correct_value = -self.mapping["change_steps"][distortion][parameter][polarity] * (list(reversed(self.mapping[distortion][parameter][polarity])).index(change) + 1)

                            to_be_regressed_param_column = results[change][f"{distortion}_change_strength"][:, list(self.mapping[distortion].keys()).index(parameter)]
                            correct_list = [correct_value] * len(to_be_regressed_param_column)
                            correct_list = torch.Tensor(correct_list)
                            logging.debug(f"{distortion}\t{parameter}\t{list(self.mapping[distortion].keys()).index(parameter)}\t{polarity}\t{change}\t{correct_value}\tregress\t{distortion}_change_strength")
                            change_regress_losses_step[distortion].append(self.mseloss(to_be_regressed_param_column, correct_list.to(self.device)))

                    if self.change_class:
                        for change in self.mapping[distortion][parameter][polarity]:
                            correct_list = [list(self.mapping[distortion].keys()).index(parameter)] * len(results[change][f"{distortion}_change_class"])
                            correct_list = torch.Tensor(correct_list).long()
                            logging.debug(f"{distortion}\t{parameter}\t{list(self.mapping[distortion].keys()).index(parameter)}\t{polarity}\t{change}\tclass\t{distortion}_change_class")
                            change_class_losses_step[distortion].append(self.celoss(results[change][f"{distortion}_change_class"], correct_list.to(self.device)))

        # balance losses
        ranking_loss: torch.Tensor = 0
        change_regress_loss: torch.Tensor = 0
        change_class_loss: torch.Tensor = 0

        for distortion in ["styles", "technical", "composition"]:
            ranking_loss += sum(ranking_losses_step[distortion])
            change_regress_loss += sum(change_regress_losses_step[distortion])
            change_class_loss += sum(change_class_losses_step[distortion])

        resulting_losses = dict()

        if self.scores is not None:
            resulting_losses["ranking_loss"] = h(ranking_loss, self.T)

        if self.change_regress:
            resulting_losses["change_regress_loss"] = h(change_regress_loss, self.T)

        if self.change_class:
            resulting_losses["change_class_loss"] = h(change_class_loss, self.T)

        loss = sum(resulting_losses)
        return loss

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            [{"params": self.parameters(), "lr": self.lr}],
            momentum=0.9,
            weight_decay=0.00004,
        )
        return {"optimizer": optimizer, "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=self.lr_patience), "monitor": "val"}

    def training_step(self, batch):
        loss = self._calc_loss(batch)

        tensorboard_logs = {"loss": {"train": loss}}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch):
        return {"val_loss": self._calc_loss(batch)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def prepare_data(self):
        self.train_ds = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
        self.val_ds = SSPexels(file_list_path="/workspace/dataset_processing/val_set.txt", mapping=mapping)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=40)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=40)
