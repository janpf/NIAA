import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

sys.path.insert(0, ".")

from model.datasets import FileListDistorted

print("starting")
print("loading dataset")
dataset = FileListDistorted(["/workspace/analysis/demo/bee.jpg"])

print("iterating")
dataloader = torch.utils.data.DataLoader(dataset)

for i, data in enumerate(dataloader):
    for c in range(1, data["num_corrs"] + 1):
        for s in range(1, 6):
            img: Image.Image = transforms.ToPILImage()(data[f"img{c}-{s}"][0])
            img.save(f"/workspace/analysis/demo/bee{data[f'corr_{c}']}-{s}.jpg")
