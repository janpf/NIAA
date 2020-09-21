import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnext101_32x8d


class CheckpointModule(nn.Module):
    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        return checkpoint_sequential(self.module, self.num_segments, *inputs)


class SSMTIA(nn.Module):
    def __init__(self, base_model_name: str, mapping, pretrained: bool = True, fix_features: bool = False):
        super(SSMTIA, self).__init__()

        self.base_model_name = base_model_name

        if base_model_name == "mobilenet":
            base_model = mobilenet_v2(pretrained=pretrained)
            self.feature_count = 1280

            features = base_model.features

        elif base_model_name == "resnext":
            base_model = resnext101_32x8d(pretrained=pretrained)
            # fmt: off
            features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            )
            # fmt: on
            self.feature_count = 2048

        else:
            raise NotImplementedError()

        if fix_features:
            for param in features.parameters():
                param.requires_grad = False

        self.mapping = mapping
        self.features = CheckpointModule(module=features, num_segments=len(features))

        # "self.classifiers"
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

        self.style_change_strength = nn.Sequential(
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

        nn.init.xavier_uniform(self.styles_score[1].weight)
        nn.init.xavier_uniform(self.technical_score[1].weight)
        nn.init.xavier_uniform(self.composition_score[1].weight)

        nn.init.xavier_uniform(self.style_change_strength[1].weight)
        nn.init.xavier_uniform(self.technical_change_strength[1].weight)
        nn.init.xavier_uniform(self.composition_change_strength[1].weight)

    def forward(self, x: torch.Tensor):
        if self.base_model_name == "resnext":
            x = torch.flatten(x, 1)

        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        s_s = self.styles_score(x)
        t_s = self.technical_score(x)
        c_s = self.composition_score(x)

        s_c_s = self.style_change_strength(x)
        t_c_s = self.technical_change_strength(x)
        c_c_s = self.composition_change_strength(x)

        return {"styles_score": s_s, "technical_score": t_s, "composition_score": c_s, "styles_change_strength": s_c_s, "technical_change_strength": t_c_s, "composition_change_strength": c_c_s}
