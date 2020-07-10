import torch
import torchvision.transforms as transforms

import sys

sys.path.insert(0, ".")

from model.datasets import Pexels

print("starting")
transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

print("loading dataset")
dataset = Pexels(
    file_list_path="/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/train_set.txt",
    original_present=False,
    compare_opposite_polarity=True,
    available_parameters=["brightness", "contrast", "exposure", "highlights", "saturation", "shadows", "temperature", "tint", "vibrance"],
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
