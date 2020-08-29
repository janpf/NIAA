import torch
from torchvision.models.mobilenet import mobilenet_v2


class SSMTIA(torch.nn.Module):
    def __init__(self, base_model_name: str, mapping):
        super(SSMTIA, self).__init__()

        if base_model_name == "mobilenet":
            base_model = mobilenet_v2(pretrained=True)
        else:
            raise NotImplementedError()

        self.mapping = mapping
        self.features = base_model.features

        # "self.classifiers"
        # fmt: off
        self.style_score = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=1),
            torch.nn.Sigmoid())
        self.technical_score = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=1),
            torch.nn.Sigmoid())
        self.composition_score = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=1),
            torch.nn.Sigmoid())

        self.style_change_strength = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=len(self.mapping["styles"])),
            torch.nn.Tanh())
        self.technical_change_strength = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=len(self.mapping["technical"])),
            torch.nn.Sigmoid())
        self.composition_change_strength = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=len(self.mapping["composition"])),
            torch.nn.Tanh())
        # fmt: on

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        s_s = self.style_score(x)
        t_s = self.technical_score(x)
        c_s = self.composition_score(x)

        s_c_s = self.style_change_strength(x)
        t_c_s = self.technical_change_strength(x)
        c_c_s = self.composition_change_strength(x)

        return {"styles_score": s_s, "technical_score": t_s, "composition_score": c_s, "styles_change_strength": s_c_s, "technical_change_strength": t_c_s, "composition_change_strength": c_c_s}
