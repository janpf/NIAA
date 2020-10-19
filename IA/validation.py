import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import redis
import torch
from torch import nn

sys.path[0] = "/workspace"
from IA.dataset import SSPexelsSmall as SSPexels
from IA.IA import IA
from IA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

margin = dict()
margin["styles"] = 0.2
margin["technical"] = 0.2
margin["composition"] = 0.2

test_file = "/workspace/dataset_processing/val_set.txt"
models_path = "/scratch/ckpts/IA"
out_file = "/workspace/analysis/IA/vals.csv"
batch_size = 5


r = redis.Redis(host="command-and-control")

available_models = sorted([str(p) for p in Path(models_path).glob("**/*.pth")])
logging.info(f"found models: {available_models}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(out_file)

logging.info("already done:")
logging.info(str(df))

models_to_validate = [p for p in available_models if not df.path.str.contains(p).any()]

logging.info(f"going to do {len(models_to_validate)}:")
logging.info(models_to_validate)

logging.info("creating datasets")
# datasets
SSPexels_test = SSPexels(file_list_path=test_file, mapping=mapping, return_file_name=True)
Pexels_test = torch.utils.data.DataLoader(SSPexels_test, batch_size=batch_size, drop_last=False, num_workers=40)
logging.info("datasets created")

for m in models_to_validate:
    logging.info(f"validating {m}")

    if not m in [p for p in available_models if not pd.read_csv(out_file).path.str.contains(p).any()]:
        continue

    if r.setnx(m, "working") == 0:
        logging.info("is already validating")
        continue
    else:
        r.expire(m, 60 * 60 * 5)

    scores: str = None
    if "scores-one" in m:
        scores = "one"
    elif "scores-three" in m:
        scores = "three"

    change_regress = False
    if "change_regress" in m:
        change_regress = True

    change_class = False
    if "change_class" in m:
        change_class = True

    logging.info(f"score:{scores}, change_regress:{change_regress}, change_class:{change_class}")
    logging.info("loading model")
    ia = IA(scores=scores, change_regress=change_regress, change_class=change_class, mapping=mapping, margin=margin).to(device)
    ia.load_state_dict(torch.load(m))
    ia.eval()

    losses_list = []
    for i, data in enumerate(Pexels_test):
        logging.info(f"{i}/{len(Pexels_test)}")
        with torch.no_grad():
            losses = ia.calc_loss(data)
        loss = sum([v for _, v in losses.items()])
        losses_list.append(loss)
    loss = sum(losses_list).item()

    with open(out_file, "a") as f:
        f.write(f"{m},{loss}")
