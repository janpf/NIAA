import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging

sys.path[0] = "/workspace"
from IA2NIMA.dataset import AVA


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

# datasets
Pexels_train_loader = DataLoader(AVA(mode="train"), batch_size=200, shuffle=True, drop_last=True, num_workers=200)

for i, data in enumerate(Pexels_train_loader):
    logging.info(f"{i}/{len(Pexels_train_loader)}")
