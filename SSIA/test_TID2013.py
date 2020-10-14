import logging
import sys
from pathlib import Path

import torch
import torch.utils

sys.path[0] = "/workspace"
from SSIA.dataset import TID2013
from SSIA.SSIA import SSIA
from SSIA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

model_path = "/scratch/ckpts/SSIA-BAL/pexels/mobilenet/0.0001/completely/epoch-2.pth"

if "mobilenet" in model_path:
    base_model = "mobilenet"
else:
    base_model = "resnext"

out_file = f"/workspace/analysis/not_uploaded/SSIA_{base_model}_test_scores_TID2013.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")
ssia = SSIA(base_model, mapping, pretrained=False).to(device)
ssia.load_state_dict(torch.load(model_path))
ssia.eval()

logging.info("creating datasets")
# datasets
Pexels_test = torch.utils.data.DataLoader(TID2013(), batch_size=100, drop_last=False, num_workers=50)
logging.info("datasets created")


out_file = open(out_file, "w")
out_file.write("img;scores\n")


logging.info("testing")

for i, data in enumerate(Pexels_test):
    logging.info(f"{i}/{len(Pexels_test)}")

    for key in data.keys():
        if key != "img":
            continue

        img = data[key].to(device)
        with torch.no_grad():
            out = ssia(img)
        result_dicts = [dict() for _ in range(len(data["path"]))]

        for k in out.keys():
            for i in range(len(data["path"])):
                result_dicts[i][k] = out[k].tolist()[i]

        for p, s in zip(data["path"], result_dicts):
            out_file.write(f"{p};{s}\n")

out_file.close()
