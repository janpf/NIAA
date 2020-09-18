import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from SSMTIA.dataset import SSPexelsNonTar as SSPexels
from SSMTIA.losses import PerfectLoss, EfficientRankingLoss, h
from SSMTIA.SSMTIA import SSMTIA
from SSMTIA.utils import mapping

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument("--conv_base_lr", type=float, default=0.00001)
parser.add_argument("--dense_lr", type=float, default=0.0001)
parser.add_argument("--styles_margin", type=float)
parser.add_argument("--technical_margin", type=float)
parser.add_argument("--composition_margin", type=float)
parser.add_argument("--base_model", type=str)
parser.add_argument("--lr_decay_rate", type=float, default=0.95)
parser.add_argument("--lr_decay_freq", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=58)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--fix_features", action="store_true")

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/SSMTIA/pexels/")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/SSMTIA/pexels/")
parser.add_argument("--warm_start", action="store_true")
parser.add_argument("--warm_start_epoch", type=int, default=0)
parser.add_argument("--early_stopping_patience", type=int, default=5)

config = parser.parse_args()

config.log_dir = str(Path(config.log_dir) / config.base_model)
config.ckpt_path = str(Path(config.ckpt_path) / config.base_model)

if config.fix_features:
    config.log_dir = str(Path(config.log_dir) / "fix_features")
    config.ckpt_path = str(Path(config.ckpt_path) / "fix_features")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")
ssmtia = SSMTIA(config.base_model, mapping, fix_features=config.fix_features).to(device)

# loading checkpoints, ... or not
if config.warm_start:
    logging.info("loading checkpoint")
    # Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    ssmtia.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch-{config.warm_start_epoch}.pth")))
    logging.info(f"Successfully loaded model epoch-{config.warm_start_epoch}.pth")
else:
    logging.info("starting cold")
    if Path(config.ckpt_path).exists():
        raise "model already trained, but cold training was used"

logging.info("setting learnrates")
conv_base_lr = config.conv_base_lr
dense_lr = config.dense_lr

if config.warm_start:
    for epoch in range(config.warm_start_epoch):
        # exponential learning rate decay:
        conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
        dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)

optimizer = torch.optim.RMSprop(
    [
        {"params": ssmtia.features.parameters(), "lr": conv_base_lr},
        {"params": ssmtia.styles_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.technical_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.composition_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.style_change_strength.parameters(), "lr": dense_lr},
        {"params": ssmtia.technical_change_strength.parameters(), "lr": dense_lr},
        {"params": ssmtia.composition_change_strength.parameters(), "lr": dense_lr},
    ],
    momentum=0.9,
    weight_decay=0.00004,
)
scaler = torch.cuda.amp.GradScaler()

# counting parameters
param_num = 0
for param in ssmtia.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"trainable params: {(param_num / 1e6):.2f} million")

logging.info("creating datasets")
# datasets
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
SSPexels_val = SSPexels(file_list_path="/workspace/dataset_processing/val_set.txt", mapping=mapping)

Pexels_train_loader = torch.utils.data.DataLoader(SSPexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
Pexels_val_loader = torch.utils.data.DataLoader(SSPexels_val, batch_size=config.val_batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers)
logging.info("datasets created")
# losses
erloss = EfficientRankingLoss()
ploss = PerfectLoss()
mseloss = nn.MSELoss()


def step(batch, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranking_losses_step: Dict[str, List] = dict()
    change_losses_step: Dict[str, List] = dict()
    perfect_losses_step = []

    for distortion in ["styles", "technical", "composition"]:
        ranking_losses_step[distortion] = []
        change_losses_step[distortion] = []

    original = ssmtia(batch["original"].to(device))
    for distortion in ["styles", "technical", "composition"]:
        for parameter in mapping[distortion]:
            for polarity in mapping[distortion][parameter]:
                results = {}
                for change in mapping[distortion][parameter][polarity]:
                    results[change] = ssmtia(batch[change].to(device))

                ranking_losses_step[distortion].append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score", margin=margin[distortion]))

                for change in mapping[distortion][parameter][polarity]:
                    if polarity == "pos":
                        correct_value = mapping["change_steps"][distortion][parameter][polarity] * (mapping[distortion][parameter][polarity].index(change) + 1)
                    if polarity == "neg":
                        correct_value = -mapping["change_steps"][distortion][parameter][polarity] * (list(reversed(mapping[distortion][parameter][polarity])).index(change) + 1)
                    correct_matrix = torch.zeros(batch_size, len(mapping[distortion]))
                    correct_matrix[:, list(mapping[distortion].keys()).index(parameter)] = correct_value

                    change_losses_step[distortion].append(mseloss(results[change][f"{distortion}_change_strength"], correct_matrix.to(device)))

    for distortion in ["styles", "technical", "composition"]:
        perfect_losses_step.append(ploss(original[f"{distortion}_score"]))

    # balance losses
    ranking_loss: torch.Tensor = 0
    change_loss: torch.Tensor = 0
    perfect_loss: torch.Tensor = h(sum(perfect_losses_step))

    for distortion in ["styles", "technical", "composition"]:
        ranking_loss += h(sum(ranking_losses_step[distortion]))
        change_loss += h(sum(change_losses_step[distortion]))

    return ranking_loss, change_loss, perfect_loss


if config.warm_start:
    g_step = config.warm_start_epoch * len(Pexels_train_loader)
else:
    g_step = 0

logging.info("start training")
for epoch in range(config.warm_start_epoch, config.epochs):
    for i, data in enumerate(Pexels_train_loader):
        logging.info(f"batch loaded: step {i}")

        optimizer.zero_grad()
        # forward pass + loss calculation
        with torch.cuda.amp.autocast():
            ranking_loss_batch, change_loss_batch, perfect_loss_batch = step(data, config.train_batch_size)
            loss: torch.Tensor = ranking_loss_batch + change_loss_batch + perfect_loss_batch

        writer.add_scalar("loss/train/balanced/ranking", ranking_loss_batch.data, g_step)
        writer.add_scalar("loss/train/balanced/change", change_loss_batch.data, g_step)
        writer.add_scalar("loss/train/balanced/perfect", perfect_loss_batch.data, g_step)
        writer.add_scalar("loss/train/balanced/overall", loss.data, g_step)

        logging.info(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training loss: {loss.data[0]}")

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
            writer.add_histogram("score/styles/weight", ssmtia.styles_score[1].weight, g_step)
            writer.add_histogram("score/styles/bias", ssmtia.styles_score[1].bias, g_step)

            writer.add_histogram("score/technical/weight", ssmtia.technical_score[1].weight, g_step)
            writer.add_histogram("score/technical/bias", ssmtia.technical_score[1].bias, g_step)

            writer.add_histogram("score/composition/weight", ssmtia.composition_score[1].weight, g_step)
            writer.add_histogram("score/composition/bias", ssmtia.composition_score[1].bias, g_step)

        g_step += 1
        logging.info("waiting for new batch")

    # exponential learning rate decay:
    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)

    optimizer = torch.optim.RMSprop(
        [
            {"params": ssmtia.features.parameters(), "lr": conv_base_lr},
            {"params": ssmtia.styles_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.technical_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.composition_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.style_change_strength.parameters(), "lr": dense_lr},
            {"params": ssmtia.technical_change_strength.parameters(), "lr": dense_lr},
            {"params": ssmtia.composition_change_strength.parameters(), "lr": dense_lr},
        ],
        momentum=0.9,
        weight_decay=0.00004,
    )

    logging.info("saving!")
    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    torch.save(ssmtia.state_dict(), str(Path(config.ckpt_path) / f"epoch-{epoch + 1}.pth"))

logging.info("Training complete!")
writer.close()
