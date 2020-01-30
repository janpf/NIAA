import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--csv', type=str, help='csv file')
parser.add_argument('--image', type=str, help='path to a single image')
parser.add_argument('--out', type=str, help='dest for images with predicted score')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--vis', action='store_true', help='visualization')
args = parser.parse_args()
img = args.image

if not os.path.exists(args.out):
    os.makedirs(args.out)

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

try:
    model.load_state_dict(torch.load(args.model))
    print('successfully loaded model')
except:
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])

df = pd.read_csv(args.csv, header=None)
mean, std = 0.0, 0.0

im = Image.open(img)
imt = test_transform(im)
imt = imt.unsqueeze(dim=0)
imt = imt.to(device)

with torch.no_grad():
    out = model(imt)
out = out.view(10, 1)

for j, e in enumerate(out, 1):
    mean += j * e
for k, e in enumerate(out, 1):
    std += (e * (k - mean) ** 2) ** (0.5)

gt = df[df[0] == int(Path(img).stem)].to_numpy()[:, 1:].reshape(10, 1)
gt_mean = 0.0
for l, e in enumerate(gt, 1):
    gt_mean += l * e

print(img.split('.')[0] + ' mean: %.3f | std: %.3f | GT: %.3f' % (mean, std, gt_mean))

if args.vis:
    plt.imshow(im)
    plt.axis('off')
    plt.title('%.3f (%.3f)' % (mean, gt_mean))
    plt.savefig(Path(args.out) / Path(img).stem / '_predicted.png')
