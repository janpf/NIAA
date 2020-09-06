import argparse
import sys
from pathlib import Path
import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from SSMTIA.dataset import SSPexelsDummy as SSPexels
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
parser.add_argument("--train_batch_size", type=int, default=10)
parser.add_argument("--val_batch_size", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)

# misc
parser.add_argument("--log_dir", type=str, default="/tmp/pexels")
parser.add_argument("--ckpt_path", type=str, default="/tmp/pexels")
parser.add_argument("--warm_start", action="store_true")
parser.add_argument("--warm_start_epoch", type=int, default=0)
parser.add_argument("--early_stopping_patience", type=int, default=5)

config = parser.parse_args()

config.log_dir = config.log_dir + config.base_model
config.ckpt_path = config.ckpt_path + config.base_model

margin = dict()
margin["styles"] = config.styles_margin
margin["technical"] = config.technical_margin
margin["composition"] = config.composition_margin

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

Path(config.log_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=config.log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")
ssmtia = SSMTIA(config.base_model, mapping).to(device)
logging.info("using half precision")
ssmtia.half()

logging.info("setting learnrates")
conv_base_lr = config.conv_base_lr
dense_lr = config.dense_lr

optimizer = torch.optim.RMSprop(
    [
        {"params": ssmtia.features.parameters(), "lr": conv_base_lr},
        {"params": ssmtia.style_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.technical_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.composition_score.parameters(), "lr": dense_lr},
        {"params": ssmtia.style_change_strength.parameters(), "lr": dense_lr},
        {"params": ssmtia.technical_change_strength.parameters(), "lr": dense_lr},
        {"params": ssmtia.composition_change_strength.parameters(), "lr": dense_lr},
    ],
    momentum=0.9,
    weight_decay=0.00004,
)

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
mseloss = torch.nn.MSELoss()


def step(batch, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranking_losses_step = []
    change_losses_step = []
    perfect_losses_step = []

    original = ssmtia(batch["original"].to(device))
    for distortion in ["styles", "technical", "composition"]:
        for parameter in mapping[distortion]:
            for polarity in mapping[distortion][parameter]:
                results = {}
                for change in mapping[distortion][parameter][polarity]:
                    results[change] = ssmtia(batch[change].to(device))

                ranking_losses_step.append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score", margin=margin[distortion]))

                for change in mapping[distortion][parameter][polarity]:
                    if polarity == "pos":
                        correct_value = mapping["change_steps"][distortion][parameter][polarity] * (mapping[distortion][parameter][polarity].index(change) + 1)
                    if polarity == "neg":
                        correct_value = -mapping["change_steps"][distortion][parameter][polarity] * (list(reversed(mapping[distortion][parameter][polarity])).index(change) + 1)
                    correct_matrix = torch.zeros(batch_size, len(mapping[distortion]))
                    correct_matrix[:, list(mapping[distortion].keys()).index(parameter)] = correct_value

                    change_losses_step.append(mseloss(results[change][f"{distortion}_change_strength"].float(), correct_matrix.to(device)))

    for distortion in ["styles", "technical", "composition"]:
        perfect_losses_step.append(ploss(original[f"{distortion}_score"]))

    return sum(ranking_losses_step), sum(change_losses_step), sum(perfect_losses_step)


g_step = 0

lowest_val_loss = float("inf")
val_not_improved = 0
logging.info("start training")
for epoch in range(config.warm_start_epoch, config.epochs):
    for i, data in enumerate(Pexels_train_loader):
        logging.info(f"batch loaded: step {i}")
        optimizer.zero_grad()

        # forward pass + loss calculation
        ranking_loss_batch, change_loss_batch, perfect_loss_batch = step(data, config.train_batch_size)

        writer.add_scalar("loss_ranking/train", ranking_loss_batch.data, g_step)
        writer.add_scalar("loss_change/train", change_loss_batch.data, g_step)
        writer.add_scalar("loss_perfect/train", perfect_loss_batch.data, g_step)
        writer.add_scalar("loss_overall/train", sum([ranking_loss_batch, change_loss_batch, perfect_loss_batch]).data[0], g_step)

        # scale losses
        ranking_loss_batch = h(ranking_loss_batch)
        change_loss_batch = h(change_loss_batch)
        perfect_loss_batch = h(perfect_loss_batch)
        loss = ranking_loss_batch + change_loss_batch + perfect_loss_batch

        writer.add_scalar("loss_scaled_ranking/train", ranking_loss_batch.data, g_step)
        writer.add_scalar("loss_scaled_change/train", change_loss_batch.data, g_step)
        writer.add_scalar("loss_scaled_perfect/train", perfect_loss_batch.data, g_step)
        writer.add_scalar("loss_scaled_overall/train", loss.data, g_step)

        logging.info(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training loss: {loss.data[0]}")

        # optimizing
        loss.backward()
        ssmtia.float()  # https://stackoverflow.com/a/58622937/6388328
        optimizer.step()
        ssmtia.half()

        writer.add_scalar("progress/epoch", epoch + 1, g_step)
        writer.add_scalar("progress/step", i + 1, g_step)
        writer.add_scalar("hparams/features_lr", conv_base_lr, g_step)
        writer.add_scalar("hparams/classifier_lr", dense_lr, g_step)
        writer.add_scalar("hparams/styles_margin", float(config.styles_margin), g_step)
        writer.add_scalar("hparams/technical_margin", float(config.technical_margin), g_step)
        writer.add_scalar("hparams/composition_margin", float(config.composition_margin), g_step)
        g_step += 1
        logging.info("waiting for new batch")

    # exponential learning rate decay:
    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)

    optimizer = torch.optim.RMSprop(
        [
            {"params": ssmtia.features.parameters(), "lr": conv_base_lr},
            {"params": ssmtia.style_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.technical_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.composition_score.parameters(), "lr": dense_lr},
            {"params": ssmtia.style_change_strength.parameters(), "lr": dense_lr},
            {"params": ssmtia.technical_change_strength.parameters(), "lr": dense_lr},
            {"params": ssmtia.composition_change_strength.parameters(), "lr": dense_lr},
        ],
        momentum=0.9,
        weight_decay=0.00004,
    )

logging.info("Training complete!")
writer.close()
