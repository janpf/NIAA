import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl

sys.path[0] = "/workspace"
from train_pre.IA import IA

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

parser = ArgumentParser()
# misc
parser.add_argument("--log_dir", type=str, default="/scratch/pl/train_logs/IA/pexels/")
parser.add_argument("--dirpath", type=str, default="/scratch/pl/ckpts/IA/pexels/")

parser = pl.Trainer.add_argparse_args(parser)
parser = IA.add_model_specific_args(parser)

config = parser.parse_args()

settings = [f"scores-{config.scores}"]

if config.change_regress:
    settings.append("change_regress")

if config.change_class:
    settings.append("change_class")

settings.append(f"m-{config.margin}")


for s in settings:
    config.default_root_dir = str(Path(config.dirpath) / s)

logging.info("init trainer")
trainer = pl.Trainer(auto_scale_batch_size=config.auto_scale_batch_size, auto_lr_find=config.auto_lr_find, gpus=config.gpus, benchmark=config.benchmark, precision=config.precision, fast_dev_run=config.fast_dev_run)

logging.info("loading model")
model = IA(scores=config.scores, change_regress=config.change_regress, change_class=config.change_class, margin=config.margin, lr_decay_rate=config.lr_decay_rate, lr_patience=config.lr_patience, num_worksers=config.num_workers)

logging.info("tuning model")
trainer.tune(model)


logging.info("fitting model")
trainer.fit(model)
