import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
import sys

sys.path[0] = "/workspace"
from IA.dataset import SSPexelsSmallTest
from IA.utils import mapping

import logging

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

SSPexels_train = SSPexelsSmallTest(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
Pexels_train_loader = DataLoader(SSPexels_train, batch_size=1, shuffle=True, drop_last=False, num_workers=500)

logging.info("auf gehts")
for data in Pexels_train_loader:
    logging.info(data.keys())
    for val in data["missing"]:
        logging.info(val)
