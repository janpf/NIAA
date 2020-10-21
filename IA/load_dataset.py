import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path[0] = "/workspace"
from IA.dataset import SSPexelsSmall
from IA.utils import mapping

# datasets
ds = SSPexelsSmall(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping, normalize=False)
Pexels_train_loader = DataLoader(ds, batch_size=1, shuffle=True, drop_last=True, num_workers=10)

for i, data in enumerate(Pexels_train_loader):
    for key in data.keys():
        path = f"/scratch/image_IA/{i}/{key}.jpeg"
        Path(f"/scratch/image_IA/{i}/").mkdir(exist_ok=True, parents=True)
        print(path)
        tens = torch.squeeze(data[key])
        img = transforms.ToPILImage()(tens)
        img.save(path)

    if i >= 10:
        exit()
