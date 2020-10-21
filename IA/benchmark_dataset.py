import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging

sys.path[0] = "/workspace"
from IA.dataset import SSPexelsSmall
from IA.utils import mapping


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

# datasets
ds = SSPexelsSmall(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
Pexels_train_loader = DataLoader(ds, batch_size=5, shuffle=True, drop_last=True, num_workers=40)

for i, data in enumerate(Pexels_train_loader):
    logging.info(i)
