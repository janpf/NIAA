import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path to pretrained model")
parser.add_argument("--csv", type=str, help="csv file")
parser.add_argument("--image", type=str, help="path to a single image")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images (if this is supplied '--image' is ignored)")
parser.add_argument("--out", type=str, help="dest for images with predicted score (and/or csv)")
parser.add_argument("--workers", type=int, default=4, help="number of workers")
parser.add_argument("--vis", action="store_true", help="visualization")
args = parser.parse_args()

if args.out and not os.path.exists(args.out):
    os.makedirs(args.out)

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

try:
    model.load_state_dict(torch.load(args.model))
    print("successfully loaded model")
except:
    raise

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

df = pd.read_csv(args.csv, header=None, delimiter=" ")

if args.out and args.imageFolder:
    csv_file = open(Path(args.out) / f"{Path(args.imageFolder).parts[-1]}.csv", "w")
    print("change, mean, std", file=csv_file)


def annotate_image(img):
    im = Image.open(img)
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)

    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)

    mean, std = 0.0, 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += (e * (k - mean) ** 2) ** (0.5)

    try:
        int(Path(img).stem)
    except:
        print(Path(img).stem + ", mean: %.3f, std: %.3f" % (mean, std))
        if args.out and args.imageFolder:
            print(Path(img).stem.split("_")[-1] + ", %.3f, %.3f" % (mean, std), file=csv_file)  # beautiful
        return

    gt = df[df[1] == int(Path(img).stem)].to_numpy()[:, 2:12].reshape(10, 1)
    gt = np.exp(gt) / sum(np.exp(gt))  # softmax
    gt_mean = 0.0
    for l, e in enumerate(gt, 1):
        gt_mean += l * e

    print(Path(img).stem + " mean: %.3f | std: %.3f | GT: %.3f" % (mean, std, gt_mean))

    if args.vis:
        plt.imshow(im)
        plt.axis("off")
        plt.title("%.3f (%.3f)" % (mean, gt_mean))
        plt.savefig(Path(args.out) / f"{Path(img).stem}.png")


if args.imageFolder:
    for img in Path(args.imageFolder).iterdir():
        annotate_image(str(img))
elif args.image:
    annotate_image(args.image)
else:
    raise ("What do you expect me to do?")
