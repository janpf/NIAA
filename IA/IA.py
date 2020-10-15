from typing import Dict, List, Tuple

import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from IA.losses import EfficientRankingLoss, h


class CheckpointModule(nn.Module):
    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        return checkpoint_sequential(self.module, self.num_segments, *inputs)


class IA(nn.Module):
    def __init__(self, scores: str, change_regress: bool, change_class: bool, mapping, pretrained: bool = False, fix_features: bool = False):
        super(IA, self).__init__()
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
        self.features = CheckpointModule(module=base_model.features, num_segments=len(base_model.features))

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
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["styles"])),
                nn.Tanh())
            self.technical_change_strength = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["technical"])),
                nn.Sigmoid())
            self.composition_change_strength = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=self.feature_count, out_features=len(self.mapping["composition"])),
                nn.Tanh())
            # fmt: on
            nn.init.xavier_uniform(self.styles_change_strength[1].weight)
            nn.init.xavier_uniform(self.technical_change_strength[1].weight)
            nn.init.xavier_uniform(self.composition_change_strength[1].weight)

        # predicting the class of a distortion
        if self.change_softmax:
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        if self.change_softmax:
            result_dict["styles_change_class"] = self.styles_change_class(x)
            result_dict["technical_change_class"] = self.technical_change_class(x)
            result_dict["composition_change_class"] = self.composition_change_class(x)

        return result_dict

    def calc_loss(self, batch) -> Dict[str, torch.Tensor]:
        """at this point I have basically reinvented pytorch lightning"""
        mseloss = nn.MSELoss()
        celoss = nn.CrossEntropyLoss()

        ranking_losses_step: Dict[str, List] = dict()
        change_regress_losses_step: Dict[str, List] = dict()
        change_class_losses_step: Dict[str, List] = dict()

        for distortion in ["styles", "technical", "composition"]:
            ranking_losses_step[distortion] = []
            change_regress_losses_step[distortion] = []

        original: Dict[str, torch.Tensor] = self(batch["original"].to(self.device))
        for distortion in ["styles", "technical", "composition"]:
            erloss = EfficientRankingLoss(margin=margin[distortion])
            for parameter in self.mapping[distortion]:
                for polarity in self.mapping[distortion][parameter]:
                    results = dict()
                    for change in self.mapping[distortion][parameter][polarity]:
                        results[change] = self(batch[change].to(self.device))

                    if self.scores == "one":
                        ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score="score"))
                    elif self.scores == "three":
                        ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score"))

                    if self.change_regress:
                        exit("check")
                        for change in self.mapping[distortion][parameter][polarity]:
                            if polarity == "pos":
                                correct_value = self.mapping["change_steps"][distortion][parameter][polarity] * (self.mapping[distortion][parameter][polarity].index(change) + 1)
                            if polarity == "neg":
                                correct_value = -self.mapping["change_steps"][distortion][parameter][polarity] * (list(reversed(self.mapping[distortion][parameter][polarity])).index(change) + 1)
                            to_be_regressed_param_column = results[change][f"{distortion}_change_strength"][list(self.mapping[distortion].keys()).index(parameter)]
                            correct_list = torch.Tensor([correct_value] * len(to_be_regressed_param_column))

                            change_regress_losses_step[distortion].append(mseloss(to_be_regressed_param_column, correct_list.to(self.device)))

                    if self.change_class:
                        exit("check")
                        for change in self.mapping[distortion][parameter][polarity]:
                            change_class_losses_step[distortion].append(celoss(results[change][f"{distortion}_change_class"], torch.Tensor([list(self.mapping[distortion].keys()).index(parameter)])))

        # balance losses
        ranking_loss: torch.Tensor = 0
        change_regress_loss: torch.Tensor = 0
        change_class_loss: torch.Tensor = 0

        for distortion in ["styles", "technical", "composition"]:
            ranking_loss += sum(ranking_losses_step[distortion])
            change_regress_loss += sum(change_regress_losses_step[distortion])
            change_class_loss += sum(change_class_losses_step[distortion])

        ranking_loss = h(ranking_loss)
        change_regress_loss = h(change_regress_loss)
        change_class_loss = h(change_class_loss)

        resulting_losses = dict()

        if self.scores is not None:
            resulting_losses["ranking_loss"] = ranking_loss

        if self.change_regress:
            resulting_losses["change_regress_loss"] = change_regress_loss

        if self.change_class:
            resulting_losses["change_class_loss"] = change_class_loss

        return resulting_losses
