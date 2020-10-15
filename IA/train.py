import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from IA.dataset import SSPexelsNonTar as SSPexels
from IA.losses import EfficientRankingLoss
from IA.IA import IA
from IA.utils import mapping

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument("--conv_base_lr", type=float, default=0.0001)
parser.add_argument("--dense_lr", type=float, default=0.0001)
parser.add_argument("--styles_margin", type=float, default=0.2)
parser.add_argument("--technical_margin", type=float, default=0.2)
parser.add_argument("--composition_margin", type=float, default=0.2)
parser.add_argument("--base_model", type=str)
parser.add_argument("--train_from", type=str)
parser.add_argument("--lr_decay_rate", type=float, default=0.9)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=60)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--fix_features", action="store_true")

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/IA/pexels/")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/IA/pexels/")
parser.add_argument("--warm_start", action="store_true")
parser.add_argument("--warm_start_epoch", type=int, default=0)

config = parser.parse_args()

settings = [config.base_model, str(config.conv_base_lr)]

if config.fix_features:
    settings.append("fix_features")

if config.softplus:
    settings.append("softplus")

for s in settings:
    config.log_dir = str(Path(config.log_dir) / s)
    config.ckpt_path = str(Path(config.ckpt_path) / s)

margin = dict()
margin["styles"] = config.styles_margin
margin["technical"] = config.technical_margin
margin["composition"] = config.composition_margin

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

if not config.warm_start and Path(config.log_dir).exists():
    raise "train script got restarted, although previous run exists"
Path(config.log_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=config.log_dir)

device = torch.device("cuda" if cuda.is_available() else "cpu")

logging.info("loading model")
ia = IA(mapping=mapping, fix_features=config.fix_features).to(device)

# loading checkpoints, ... or not
if config.warm_start:
    logging.info("loading checkpoint")
    # Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    ia.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch-{config.warm_start_epoch}.pth")))
    logging.info(f"Successfully loaded model epoch-{config.warm_start_epoch}.pth")

elif config.train_from:
    logging.info("loading from")
    ia.load_state_dict(torch.load(config.train_from))

else:
    logging.info("starting cold")
    if Path(config.ckpt_path).exists():
        raise "model already trained, but cold training was used"

logging.info("setting learnrates")

# fmt:off
optimizer = optim.RMSprop(
    [
        {"params": ia.features.parameters(), "lr": config.conv_base_lr},
        {"params": ia.styles_score.parameters(), "lr": config.dense_lr},
        {"params": ia.technical_score.parameters(), "lr": config.dense_lr},
        {"params": ia.composition_score.parameters(), "lr": config.dense_lr}],
        momentum=0.9,
        weight_decay=0.00004,
)
lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: config.lr_decay_rate ** (epoch+1))
# fmt:on
scaler = cuda.amp.GradScaler()

# counting parameters
param_num = 0
for param in ia.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"trainable params: {(param_num / 1e6):.2f} million")

logging.info("creating datasets")
# datasets
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
Pexels_train_loader = DataLoader(SSPexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
logging.info("datasets created")

if config.warm_start:
    g_step = config.warm_start_epoch * len(Pexels_train_loader)
    for epoch in range(config.warm_start_epoch):
        lr_scheduler.step()
else:
    g_step = 0


logging.info("start training")
for epoch in range(config.warm_start_epoch, config.epochs):
    for i, data in enumerate(Pexels_train_loader):
        logging.info(f"batch loaded: step {i}")

        optimizer.zero_grad()
        # forward pass + loss calculation
        with cuda.amp.autocast():
            losses = ia.calc_loss(data)

        writer.add_scalar("loss/train/balanced/ranking", loss.item(), g_step)

        logging.info(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training loss: {loss.item()}")

        # optimizing
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("progress/epoch", epoch + 1, g_step)
        writer.add_scalar("progress/step", i + 1, g_step)
        writer.add_scalar("hparams/lr/features", optimizer.param_groups[0]["lr"], g_step)
        writer.add_scalar("hparams/lr/classifier", optimizer.param_groups[1]["lr"], g_step)
        writer.add_scalar("hparams/margin/styles", float(config.styles_margin), g_step)
        writer.add_scalar("hparams/margin/technical", float(config.technical_margin), g_step)
        writer.add_scalar("hparams/margin/composition", float(config.composition_margin), g_step)

        if g_step % 100 == 0:
            exit("fixme")
            writer.add_histogram("score/styles/weight", ia.styles_score[1].weight, g_step)
            writer.add_histogram("score/styles/bias", ia.styles_score[1].bias, g_step)

            writer.add_histogram("score/technical/weight", ia.technical_score[1].weight, g_step)
            writer.add_histogram("score/technical/bias", ia.technical_score[1].bias, g_step)

            writer.add_histogram("score/composition/weight", ia.composition_score[1].weight, g_step)
            writer.add_histogram("score/composition/bias", ia.composition_score[1].bias, g_step)

        g_step += 1
        logging.info("waiting for new batch")

    # learning rate decay:
    lr_scheduler.step()

    logging.info("saving!")
    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    torch.save(ia.state_dict(), str(Path(config.ckpt_path) / f"epoch-{epoch + 1}.pth"))

logging.info("Training complete!")
writer.close()
