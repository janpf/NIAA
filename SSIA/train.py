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
from SSIA.dataset import SSPexelsNonTar as SSPexels
from SSIA.losses import EfficientRankingLoss, h
from SSIA.SSIA import SSIA
from SSIA.utils import mapping

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument("--conv_base_lr", type=float, default=0.0001)
parser.add_argument("--dense_lr", type=float, default=0.0001)
parser.add_argument("--styles_margin", type=float)
parser.add_argument("--technical_margin", type=float)
parser.add_argument("--composition_margin", type=float)
parser.add_argument("--base_model", type=str)
parser.add_argument("--train_from", type=str)
parser.add_argument("--lr_decay_rate", type=float, default=0.95)
parser.add_argument("--lr_decay_freq", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=60)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--fix_features", action="store_true")

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/SSIA/pexels/")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/SSIA/pexels/")
parser.add_argument("--warm_start", action="store_true")
parser.add_argument("--warm_start_epoch", type=int, default=0)

config = parser.parse_args()

config.log_dir = str(Path(config.log_dir) / config.base_model)
config.ckpt_path = str(Path(config.ckpt_path) / config.base_model)

if config.fix_features:
    config.log_dir = str(Path(config.log_dir) / "fix_features")
    config.ckpt_path = str(Path(config.ckpt_path) / "fix_features")
else:
    config.log_dir = str(Path(config.log_dir) / "completely")
    config.ckpt_path = str(Path(config.ckpt_path) / "completely")
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
ssia = SSIA(config.base_model, mapping, fix_features=config.fix_features).to(device)

# loading checkpoints, ... or not
if config.warm_start:
    logging.info("loading checkpoint")
    # Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    ssia.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch-{config.warm_start_epoch}.pth")))
    logging.info(f"Successfully loaded model epoch-{config.warm_start_epoch}.pth")

elif config.train_from:
    logging.info("loading from")
    ssia.load_state_dict(torch.load(config.train_from))

else:
    logging.info("starting cold")
    if Path(config.ckpt_path).exists():
        raise "model already trained, but cold training was used"

logging.info("setting learnrates")
conv_base_lr = config.conv_base_lr
dense_lr = config.dense_lr

# fmt:off
optimizer = optim.RMSprop(
    [
        {"params": ssia.features.parameters(), "lr": conv_base_lr},
        {"params": ssia.styles_score.parameters(), "lr": dense_lr},
        {"params": ssia.technical_score.parameters(), "lr": dense_lr},
        {"params": ssia.composition_score.parameters(), "lr": dense_lr}],
        momentum=0.9,
        weight_decay=0.00004,
)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
# fmt:on
scaler = cuda.amp.GradScaler()

# counting parameters
param_num = 0
for param in ssia.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"trainable params: {(param_num / 1e6):.2f} million")

logging.info("creating datasets")
# datasets
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
SSPexels_val = SSPexels(file_list_path="/workspace/dataset_processing/val_set.txt", mapping=mapping)

Pexels_train_loader = DataLoader(SSPexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
Pexels_val_loader = DataLoader(SSPexels_val, batch_size=config.val_batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers)
logging.info("datasets created")

if config.warm_start:
    g_step = config.warm_start_epoch * len(Pexels_train_loader)
    for epoch in range(config.warm_start_epoch):
        lr_scheduler.step()
else:
    g_step = 0


def step(batch) -> torch.Tensor:
    ranking_losses_step: Dict[str, List] = dict()

    for distortion in ["styles", "technical", "composition"]:
        ranking_losses_step[distortion] = []

    original: Dict[str, torch.Tensor] = ssia(batch["original"].to(device))
    for distortion in ["styles", "technical", "composition"]:
        erloss = EfficientRankingLoss(margin=margin[distortion])
        for parameter in mapping[distortion]:
            for polarity in mapping[distortion][parameter]:
                results = dict()
                for change in mapping[distortion][parameter][polarity]:
                    results[change] = ssia(batch[change].to(device))

                ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score"))

    # balance losses
    ranking_loss: torch.Tensor = 0

    for distortion in ["styles", "technical", "composition"]:
        ranking_loss += h(sum(ranking_losses_step[distortion]))

    return ranking_loss


logging.info("start training")
for epoch in range(config.warm_start_epoch, config.epochs):
    for i, data in enumerate(Pexels_train_loader):
        logging.info(f"batch loaded: step {i}")

        optimizer.zero_grad()
        # forward pass + loss calculation
        with cuda.amp.autocast():
            loss = step(data)

        writer.add_scalar("loss/train/balanced/ranking", loss.item(), g_step)

        logging.info(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training loss: {loss.item()}")

        # optimizing
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("progress/epoch", epoch + 1, g_step)
        writer.add_scalar("progress/step", i + 1, g_step)
        writer.add_scalar("hparams/lr/features", conv_base_lr, g_step)
        writer.add_scalar("hparams/lr/classifier", dense_lr, g_step)
        writer.add_scalar("hparams/margin/styles", float(config.styles_margin), g_step)
        writer.add_scalar("hparams/margin/technical", float(config.technical_margin), g_step)
        writer.add_scalar("hparams/margin/composition", float(config.composition_margin), g_step)

        if g_step % 100 == 0:
            writer.add_histogram("score/styles/weight", ssia.styles_score[1].weight, g_step)
            writer.add_histogram("score/styles/bias", ssia.styles_score[1].bias, g_step)

            writer.add_histogram("score/technical/weight", ssia.technical_score[1].weight, g_step)
            writer.add_histogram("score/technical/bias", ssia.technical_score[1].bias, g_step)

            writer.add_histogram("score/composition/weight", ssia.composition_score[1].weight, g_step)
            writer.add_histogram("score/composition/bias", ssia.composition_score[1].bias, g_step)

        g_step += 1
        logging.info("waiting for new batch")

    # exponential learning rate decay:
    lr_scheduler.step()

    logging.info("saving!")
    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    torch.save(ssia.state_dict(), str(Path(config.ckpt_path) / f"epoch-{epoch + 1}.pth"))

logging.info("Training complete!")
writer.close()
