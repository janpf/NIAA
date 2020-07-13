import torch
import torchvision.transforms as transforms

import sys

sys.path.insert(0, ".")

from model.datasets import PexelsRedis

print("starting")
transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

print("loading dataset")
dataset = PexelsRedis(mode="train", transforms=transform)
print(f"loaded: {len(dataset)}")

print("iterating")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)

for i, data in enumerate(dataloader):
    print(data)
    exit()
