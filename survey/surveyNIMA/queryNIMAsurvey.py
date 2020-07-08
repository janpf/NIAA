import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import math


sys.path.insert(0, ".")
from model.NIMA import NIMA
from edit_image import parameter_range

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(Path("/scratch") / "pretrained_new.pth"), type=str, help="path to pretrained model")
parser.add_argument("--survey_csv", default=str(Path("/scratch") / "pexels" / "logs" / "survey.csv"), type=str, help="test csv file")
parser.add_argument("--original_img_dir", type=str, default="/scratch/pexels/images")
parser.add_argument("--edited_img_dir", type=str, default="/scratch/pexels/edited_images")
args = parser.parse_args()

df = pd.read_csv(args.survey_csv)  # type: pd.DataFrame
df = df[df.chosen != "error"]
df = df[df.chosen != "unsure"]
df = df.assign(leftNIMA=np.nan, rightNIMA=np.nan)

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load(args.model))
print("successfully loaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()
# fmt: off
test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])
# fmt: on


def predictImage(path: str):
    im = Image.open(path)
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    mean = 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
    return mean


for i, row in df.iterrows():
    print(f"working on {row['hashval']}")
    print(row)

    if row["parameter"] == "lcontrast":  # those are not preprocessed
        leftScore = predictImage(str(Path(args.test_images) / f"{row['hashval']}l.jpg"))
        rightScore = predictImage(str(Path(args.test_images) / f"{row['hashval']}r.jpg"))
    else:
        if math.isclose(row["leftChanges"], parameter_range[row["parameter"]]["default"]):
            leftScore = predictImage(str(Path(args.original_img_dir) / f"{row['img'].replace('/img/', '')}"))
        else:
            leftScore = predictImage(str(Path(args.edited_img_dir) / row["parameter"] / str(row["leftChanges"]) / f"{row['img'].replace('/img/', '')}"))

        if math.isclose(row["rightChanges"], parameter_range[row["parameter"]]["default"]):
            rightScore = predictImage(str(Path(args.original_img_dir) / f"{row['img'].replace('/img/', '')}"))
        else:
            rightScore = predictImage(str(Path(args.edited_img_dir) / row["parameter"] / str(row["rightChanges"]) / f"{row['img'].replace('/img/', '')}"))

    df.at[i, "leftNIMA"] = leftScore
    df.at[i, "rightNIMA"] = rightScore

df.to_csv(str(Path("/data") / "logs" / "survey_NIMA.csv"))
