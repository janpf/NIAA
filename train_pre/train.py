from argparse import ArgumentParser
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_pre import IA

parser = ArgumentParser()
# misc
parser.add_argument("--log_dir", type=str, default="/scratch/pl/train_logs/IA/pexels/")
parser.add_argument("--dirpath", type=str, default="/scratch/pl/ckpts/IA/pexels/")

model_parser = IA.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

config = parser.parse_args()

settings = [f"scores-{config.scores}"]

if config.change_regress:
    settings.append("change_regress")

if config.change_class:
    settings.append("change_class")

settings.append(f"m-{config.margin}")

for s in settings:
    config.default_root_dir = str(Path(config.dirpath) / s)


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

logging.info("init trainer")
trainer = pl.Trainer.from_argparse_args(config)
# auto_scale_batch_size=True
# auto_lr_find=True
# gpus=-1
# benchmark=True
# precision=16

# fast_dev_run=True

logging.info("loading model")
model = IA(config)

logging.info("tuning model")
trainer.tune(model)

logging.info("fitting model")
trainer.fit()
