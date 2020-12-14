import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import cuda

sys.path[0] = "/workspace"
from IA.dataset import TID2013
from IA.IA import IA
from IA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default=None)
config = parser.parse_args()


margin = dict()
margin["styles"] = 0.2
margin["technical"] = 0.2
margin["composition"] = 0.2

test_file = "/workspace/dataset_processing/test_set.txt"
out_file = "/workspace/analysis/not_uploaded/IA/TID/" + config.model_path.replace("/", ".") + ".txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scores: str = None
if "scores-one" in config.model_path:
    scores = "one"
elif "scores-three" in config.model_path:
    scores = "three"

change_regress = False
if "change_regress" in config.model_path:
    change_regress = True

change_class = False
if "change_class" in config.model_path:
    change_class = True

logging.info(f"score:{scores}, change_regress:{change_regress}, change_class:{change_class}")
logging.info("loading model")

ia = IA(scores=scores, change_regress=change_regress, change_class=change_class, mapping=mapping, margin=margin).to(device)
ia.load_state_dict(torch.load(config.model_path))
ia.eval()

logging.info("creating datasets")
# datasets
Pexels_test = torch.utils.data.DataLoader(TID2013(), batch_size=5, drop_last=False, num_workers=40)
logging.info("datasets created")

logging.info("testing")

output = []

for i, data in enumerate(Pexels_test):
    logging.info(f"{i}/{len(Pexels_test)}")

    with cuda.amp.autocast():
        with torch.no_grad():
            out = ia(data["img"].to(device))
    result_dicts = [dict() for _ in range(len(data["path"]))]

    for k in out.keys():
        for i in range(len(data["path"])):
            result_dicts[i][k] = out[k].tolist()[i]
            if len(result_dicts[i][k]) == 1:
                result_dicts[i][k] = result_dicts[i][k][0]

    for p, s in zip(data["path"], result_dicts):
        out_dict = dict()
        out_dict["img"] = p

        for k in s.keys():
            if "strength" in k or "class" in k:
                for i, param in enumerate(mapping[k.split("_")[0]].keys()):
                    out_dict[f"{k.split('_')[0]}_{param}_{k.split('_')[-1]}"] = s[k][i]
            else:
                out_dict[k] = s[k]
        output.append(out_dict)

logging.info(f"writing {out_file}")
pd.DataFrame(output).to_csv(out_file, index=False)
