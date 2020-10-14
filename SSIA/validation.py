import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch import nn

sys.path[0] = "."
from SSIA.dataset import SSPexelsNonTar as SSPexels
from SSIA.SSIA import SSIA
from SSIA.losses import EfficientRankingLoss, h
from SSIA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

margin = dict()
margin["styles"] = 0.1
margin["technical"] = 0.2
margin["composition"] = 0.2


def step(batch) -> torch.Tensor:
    ranking_losses_step: Dict[str, List] = dict()

    for distortion in ["styles", "technical", "composition"]:
        ranking_losses_step[distortion] = []

    original: Dict[str, torch.Tensor] = ssia(batch["original"].to(device))
    for distortion in ["styles", "technical", "composition"]:
        erloss = EfficientRankingLoss(margin=margin[distortion], softplus=config.softplus)
        for parameter in mapping[distortion]:
            for polarity in mapping[distortion][parameter]:
                results = dict()
                for change in mapping[distortion][parameter][polarity]:
                    results[change] = ssia(batch[change].to(device))

                ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score"))

    ranking_loss: torch.Tensor = sum([sum(ranking_losses_step[distortion]) for distortion in ["styles", "technical", "composition"]])

    return ranking_loss


test_file = "/workspace/dataset_processing/val_set.txt"
models_path = "/scratch/ckpts/SSIA"
out_file = "/workspace/analysis/SSIA/vals.csv"
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
Pexels_test = torch.utils.data.DataLoader(SSPexels_test, batch_size=batch_size, drop_last=False, num_workers=48)
logging.info("datasets created")

for m in models_to_validate:
    if "mobilenet" in m:
        base_model = "mobilenet"
    else:
        base_model = "resnext"
    logging.info(f"validating {m} as {base_model}")

    logging.info("loading model")
    ssia = SSIA(base_model, mapping, pretrained=False).to(device)
    ssia.load_state_dict(torch.load(m))
    ssia.eval()

    losses = []
    for i, data in enumerate(Pexels_test):
        logging.info(f"{i}/{len(Pexels_test)}")
        with torch.no_grad():
            losses.append(step(data))

    loss = sum(losses).data
    df = df.append({"path": m, "loss": loss}, ignore_index=True)
    df = df.sort_values(by=["path"])
    df.to_csv(out_file, index=False)
