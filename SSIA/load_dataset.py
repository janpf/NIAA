import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path[0] = "/workspace"
from SSIA.dataset import SSPexelsNonTar as SSPexels
from SSIA.utils import mapping

# datasets
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping, normalize=False)
Pexels_train_loader = DataLoader(SSPexels_train, batch_size=1, shuffle=True, drop_last=True, num_workers=10)

for i, data in enumerate(Pexels_train_loader):
    for change in data.keys():
        path = f"/scratch/image/{i}/{change}.jpeg"
        Path(f"/scratch/image/{i}/").mkdir(exist_ok=True, parents=True)
        print(path)
        tens = torch.squeeze(data[change])
        img = transforms.ToPILImage()(tens)
        img.save(path)

    if i >= 10:
        exit()
