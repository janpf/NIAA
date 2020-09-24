import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch import nn

sys.path[0] = "."
from SSMTIA.dataset import SSPexelsNonTar as SSPexels
from SSMTIA.SSMTIA import SSMTIA
from SSMTIA.losses import PerfectLoss, EfficientRankingLoss, h
from SSMTIA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

margin = dict()
margin["styles"] = 0.1
margin["technical"] = 0.2
margin["composition"] = 0.2

erloss = EfficientRankingLoss()
ploss = PerfectLoss()
mseloss = nn.MSELoss()


def step(batch, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranking_losses_step: Dict[str, List] = dict()
    change_losses_step: Dict[str, List] = dict()
    perfect_losses_step = []

    for distortion in ["styles", "technical", "composition"]:
        ranking_losses_step[distortion] = []
        change_losses_step[distortion] = []

    original = ssmtia(batch["original"].to(device))
    for distortion in ["styles", "technical", "composition"]:
        for parameter in mapping[distortion]:
            for polarity in mapping[distortion][parameter]:
                results = {}
                for change in mapping[distortion][parameter][polarity]:
                    results[change] = ssmtia(batch[change].to(device))

                ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score", margin=margin[distortion]))

                for change in mapping[distortion][parameter][polarity]:
                    if polarity == "pos":
                        correct_value = mapping["change_steps"][distortion][parameter][polarity] * (mapping[distortion][parameter][polarity].index(change) + 1)
                    if polarity == "neg":
                        correct_value = -mapping["change_steps"][distortion][parameter][polarity] * (list(reversed(mapping[distortion][parameter][polarity])).index(change) + 1)
                    correct_matrix = torch.zeros(batch_size, len(mapping[distortion]))
                    correct_matrix[:, list(mapping[distortion].keys()).index(parameter)] = correct_value

                    change_losses_step[distortion].append(mseloss(results[change][f"{distortion}_change_strength"], correct_matrix.to(device)))

    for distortion in ["styles", "technical", "composition"]:
        perfect_losses_step.append(ploss(original[f"{distortion}_score"]))

    # balance losses
    ranking_loss: torch.Tensor = 0
    change_loss: torch.Tensor = 0
    perfect_loss: torch.Tensor = h(sum(perfect_losses_step))

    for distortion in ["styles", "technical", "composition"]:
        ranking_loss += h(sum(ranking_losses_step[distortion]))
        change_loss += h(sum(change_losses_step[distortion]))

    return ranking_loss, change_loss, perfect_loss


test_file = "/workspace/dataset_processing/val_set.txt"
models_path = "/scratch/ckpts/SSMTIA"
out_file = "/workspace/analysis/SSMTIA/vals.csv"
batch_size = 5

available_models = sorted([str(p) for p in Path(models_path).glob("**/*.pth")])
logging.info(f"found models: {available_models}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(out_file)

logging.info("already done:")
logging.info(str(df))

models_to_validate = [p for p in available_models if not df.path.str.contains(p).any()]

logging.info("going to do:")
logging.info(models_to_validate)

logging.info("creating datasets")
# datasets
SSPexels_test = SSPexels(file_list_path=test_file, mapping=mapping, return_file_name=True)
Pexels_test = torch.utils.data.DataLoader(SSPexels_test, batch_size=batch_size, drop_last=False, num_workers=24)
logging.info("datasets created")


for m in models_to_validate:
    base_model = Path(m).parts[-3]
    logging.info(f"validating {m} as {base_model}")

    logging.info("loading model")
    ssmtia = SSMTIA(base_model, mapping, pretrained=False).to(device)
    ssmtia.load_state_dict(torch.load(m))
    ssmtia.eval()

    losses = []
    for i, data in enumerate(Pexels_test):
        logging.info(f"{i}/{len(Pexels_test)}")
        with torch.no_grad():
            losses.append(sum(step(data, batch_size)))

    loss = sum(losses).data
    df = df.append({"path": m, "loss": loss}, ignore_index=True)
    df.to_csv(out_file, index=False)
