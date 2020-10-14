import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path[0] = "/workspace"
from SSIA.dataset import AVA
from SSIA.utils import mapping

# datasets
# ds = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping, normalize=False)
ds = AVA(normalize=False)
Pexels_train_loader = DataLoader(ds, batch_size=1, shuffle=True, drop_last=True, num_workers=10)

for i, data in enumerate(Pexels_train_loader):
    path = f"/scratch/image_AVA/{i}.jpeg"
    Path(f"/scratch/image_AVA/").mkdir(exist_ok=True, parents=True)
    print(path)
    tens = torch.squeeze(data["img"])
    img = transforms.ToPILImage()(tens)
    img.save(path)

    if i >= 10:
        exit()
