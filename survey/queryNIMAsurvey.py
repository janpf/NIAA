import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path to pretrained model")
parser.add_argument("--survey_csv", type=str, help="test csv file")
parser.add_argument("--test_images", type=str, help="path to folder containing images")
parser.add_argument("--out", type=str, help="dest for images with predicted score")
args = parser.parse_args()

if not os.path.exists(args.out):
    os.makedirs(args.out)

base_model = models.vgg16(pretrained=False)  #  TODO check, pretrained dürfte nicht nötig sein, da eh gleich überschrieben wird
model = NIMA(base_model)

try:
    model.load_state_dict(torch.load(args.model))
    print("successfully loaded model")
except:
    raise FileNotFoundError("couldn't load pretrained model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()
# fmt: off
test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])
# fmt: on

test_imgs = [f for f in os.listdir(args.test_images)]

test_df = pd.read_csv(args.survey_csv)

for i, img in enumerate(test_imgs):
    im = Image.open(os.path.join(args.test_images, img))
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
