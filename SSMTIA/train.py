import argparse
import sys
from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from SSMTIA.dataset import SSPexels
from SSMTIA.losses import PerfectLoss, EfficientRankingLoss, h
from SSMTIA.SSMTIA import SSMTIA
from SSMTIA.utils import mapping

parser = argparse.ArgumentParser()

# input parameters
parser.add_argument("--val_imgs_count", type=int, default=500000)
parser.add_argument("--orig_present", action="store_true")

# training parameters
parser.add_argument("--conv_base_lr", type=float, default=0.0005)  # https://github.com/kentsyx/Neural-IMage-Assessment/issues/16
parser.add_argument("--dense_lr", type=float, default=0.005)  # https://github.com/kentsyx/Neural-IMage-Assessment/issues/16
parser.add_argument("--margin", type=float)
parser.add_argument("--lr_decay_rate", type=float, default=0.95)
parser.add_argument("--lr_decay_freq", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--val_batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=24)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--img_per_p_epoch", type=int, default=1000000)

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/pexels/")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/pexels/")
parser.add_argument("--dir_name", type=str, default="cold")
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--gpu_ids", type=list, default=None)
parser.add_argument("--warm_start", action="store_true")
parser.add_argument("--warm_start_epoch", type=int, default=0)
parser.add_argument("--early_stopping_patience", type=int, default=5)

config = parser.parse_args()

config.log_dir = config.log_dir + config.dir_name
config.ckpt_path = config.ckpt_path + config.dir_name


if not config.warm_start and Path(config.log_dir).exists():
    raise "train script got restarted, although previous run exists"
Path(config.log_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=config.log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing model
ssmtia = SSMTIA("mobilenet", mapping).to(device)
ssmtia.half()

# loading checkpoints, ... or not
if config.warm_start:
    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    ssmtia.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch{config.warm_start_epoch}-p_epoch{config.warm_start_p_epoch}.pth")))
    logging.info(f"Successfully loaded model epoch{config.warm_start_epoch}-p_epoch{config.warm_start_p_epoch}.pth")
else:
    logging.info("starting cold")
    if Path(config.ckpt_path).exists():
        raise "model already trained, but cold training was used"

# settings learnrates
conv_base_lr = config.conv_base_lr
dense_lr = config.dense_lr

# fmt: off
optimizer = torch.optim.RMSprop([
    {"params": ssmtia.features.parameters(), "lr": conv_base_lr, "weight_decay": 0.00004, "momentum": 0.9},
    {"params": ssmtia.classifier.parameters(), "lr": dense_lr, "weight_decay": 0.00004, "momentum": 0.9}], # FIXME, parameters
    momentum=0.9)
# fmt: on

writer.add_scalar("hparams/features_lr", config.conv_base_lr, 0)
writer.add_scalar("hparams/classifier_lr", config.dense_lr, 0)

# counting parameters
param_num = 0
for param in ssmtia.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"Trainable params: {(param_num / 1e6):.2f} million")

# datasets
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
SSPexels_val = SSPexels(file_list_path="/workspace/dataset_processing/val_set.txt", mapping=mapping)

Pexels_train_loader = torch.utils.data.DataLoader(SSPexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
Pexels_val_loader = torch.utils.data.DataLoader(SSPexels_val, batch_size=config.val_batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers)

# losses
erloss = EfficientRankingLoss()
ploss = PerfectLoss()
mseloss = torch.nn.MSELoss()


def step(batch, batch_size: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
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

                ranking_losses_step.append(erloss(original, x=results, polarity=polarity, score=f"{distortion}_score"))

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

    # TODO is everything float?
    return sum(ranking_losses_step), sum(change_losses_step), sum(perfect_losses_step)


g_step = 0
lowest_val_loss = float("inf")
val_not_improved = 0

for epoch in range(config.warm_start_epoch, config.epochs):
    for i, data in enumerate(Pexels_train_loader):
        optimizer.zero_grad()

        # forward pass + loss calculation
        ranking_loss_batch, change_loss_batch, perfect_loss_batch = step(data, config.train_batch_size)
        # scale losses
        ranking_loss_batch = h(ranking_loss_batch)
        change_loss_batch = h(change_loss_batch)
        perfect_loss_batch = h(perfect_loss_batch)
        loss = ranking_loss_batch + change_loss_batch + perfect_loss_batch

        # optimizing
        loss.backward()
        optimizer.step()

        logging.info(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training loss: {loss.data[0]:.4f}")
        g_step += 1

    # learning rate decay:
    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)

    # fmt: off
    optimizer = torch.optim.RMSprop([
        {"params": ssmtia.features.parameters(), "lr": conv_base_lr, "weight_decay": 0.00004, "momentum": 0.9},
        {"params": ssmtia.classifier.parameters(), "lr": dense_lr, "weight_decay": 0.00004, "momentum": 0.9}],
        momentum=0.9)
    # fmt: on

    logging.info("validating")

    ranking_loss = []
    change_loss = []
    perfect_loss = []

    ranking_loss_scaled = []
    change_loss_scaled = []
    perfect_loss_scaled = []

    for i, data in enumerate(Pexels_val_loader):
        with torch.no_grad():
            ranking_loss_batch, change_loss_batch, perfect_loss_batch = step(data, config.val_batch_size)

        ranking_loss.append(ranking_loss_batch.item())
        change_loss.append(change_loss_batch.item())
        perfect_loss.append(perfect_loss_batch.item())

        ranking_loss_scaled.append(h(ranking_loss_batch).item())
        change_loss_scaled.append(h(change_loss_batch).item())
        perfect_loss_scaled.append(h(perfect_loss_batch).item())

    val_counts = len(ranking_loss)
    overall_loss = (sum(ranking_loss) + sum(change_loss) + sum(perfect_loss)) / val_counts
    ranking_loss = sum(ranking_loss) / val_counts
    change_loss = sum(change_loss) / val_counts
    perfect_loss = sum(perfect_loss) / val_counts

    overall_loss_scaled = (sum(ranking_loss_scaled) + sum(change_loss_scaled) + sum(perfect_loss_scaled)) / val_counts
    ranking_loss_scaled = sum(ranking_loss_scaled) / val_counts
    change_loss_scaled = sum(change_loss_scaled) / val_counts
    perfect_loss_scaled = sum(perfect_loss_scaled) / val_counts

    # Use early stopping to monitor training
    if overall_loss < lowest_val_loss:
        lowest_val_loss = overall_loss

        logging.info("New best model! Saving...")
        Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
        torch.save(ssmtia.state_dict(), str(Path(config.ckpt_path) / f"epoch{epoch + 1}.pth"))
        # reset count
        val_not_improved = 0
    else:
        val_not_improved += 1
        if val_not_improved >= config.early_stopping_patience:
            logging.info("early stopping")

logging.info("Training complete!")
writer.close()


# TODO writer
# TODO find best batchsize
# TODO k8s files
