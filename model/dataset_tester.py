import torch
import torchvision.transforms as transforms

import sys

sys.path.insert(0, ".")

from model.datasets import Pexels

print("starting")
transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

print("loading dataset")
dataset = Pexels(
    file_list_path="/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/val_set.txt",
    original_present=True,
    available_parameters=["brightness", "contrast"],
    transforms=transform,
    orig_dir="/scratch/stud/pfister/NIAA/pexels/images",
    edited_dir="/scratch/stud/pfister/NIAA/pexels/edited_images",
)
print(f"loaded: {len(dataset)}")

print("iterating")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)

for i, data in enumerate(dataloader):
    print(data)
    exit()
