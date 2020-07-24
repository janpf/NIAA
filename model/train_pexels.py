import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms

sys.path[0] = "/workspace"

from model.datasets import PexelsRedis
from model.NIAA import NIAA, Distance_Loss, SiameseBatchAccuracy


def main(config):
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fmt: off
    Pexels_train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    Pexels_val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    # fmt: on
    base_model = models.vgg16(pretrained=True)
    model = NIAA(base_model)
    dist_loss = Distance_Loss()
    accuracy = SiameseBatchAccuracy()

    if config.warm_start:
        Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
        model.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch{config.warm_start_epoch}-p_epoch{config.warm_start_p_epoch}.pth")))
        print(f"Successfully loaded model epoch{config.warm_start_epoch}-p_epoch{config.warm_start_p_epoch}.pth")
    else:
        print("starting cold", flush=True)
        if Path(config.ckpt_path).exists():
            raise "model already trained, but cold training was used"

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr

    # fmt: off
    optimizer = torch.optim.SGD([
        {'params': model.features.parameters(), 'lr': config.conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': config.dense_lr}],
        momentum=0.9)
    # fmt: on
    writer.add_scalar("hparams/features_lr", config.conv_base_lr, 0)
    writer.add_scalar("hparams/classifier_lr", config.dense_lr, 0)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print(f"Trainable params: {(param_num / 1e6):.2f} million")

    Pexels_train = PexelsRedis(mode="train", transforms=Pexels_train_transform)
    Pexels_val = PexelsRedis(mode="val", transforms=Pexels_val_transform)

    Pexels_train_loader = torch.utils.data.DataLoader(Pexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
    Pexels_val_loader = torch.utils.data.DataLoader(Pexels_val, batch_size=config.val_batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers)

    count = 0  # for early stopping
    g_step = 0  # global step
    init_val_loss = float("inf")
    current_epoch_iter = None

    p_epochs_per_epoch = len(Pexels_train) // config.img_per_p_epoch
    print(f"every epoch is going to consist of {p_epochs_per_epoch} pseudo epochs")
    for epoch in range(config.warm_start_epoch, config.epochs):
        del current_epoch_iter
        current_epoch_iter = iter(Pexels_train_loader)
        for pseudo_epoch in range(p_epochs_per_epoch):
            p_batch_losses = []
            p_batch_acc = []
            for i, data in enumerate(current_epoch_iter):
                if i * config.train_batch_size >= config.img_per_p_epoch:
                    break

                img1 = data["img1"].to(device)
                img2 = data["img2"].to(device)
                out1, out2 = model(img1, img2, "siamese")

                optimizer.zero_grad()

                loss = dist_loss(out1, out2, epsilon=config.margin)
                p_batch_losses.append(loss.item())

                acc = accuracy(out1, out2)
                p_batch_acc.append(acc)

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch + 1}/{config.epochs} | Pseudo-epoch: {pseudo_epoch + 1}/{p_epochs_per_epoch} | Step: {i + 1}/{config.img_per_p_epoch // config.train_batch_size} | Training dist loss: {loss.data[0]:.4f} | Training acc: {acc:.4f}", flush=True)
                writer.add_scalar("progress/epoch", epoch + 1, g_step)
                writer.add_scalar("progress/p_epoch", pseudo_epoch + 1, g_step)
                writer.add_scalar("progress/step_in_p_epoch", i + 1, g_step)
                writer.add_scalar("loss/train", loss.data[0], g_step)
                writer.add_scalar("acc/train", acc, g_step)
                writer.add_scalar("hparams/features_lr", conv_base_lr, g_step)
                writer.add_scalar("hparams/classifier_lr", dense_lr, g_step)
                g_step += 1

            p_avg_loss = sum(p_batch_losses) / len(p_batch_losses)
            p_avg_acc = sum(p_batch_acc) / len(p_batch_acc)
            print(f"Pseudo-epoch {pseudo_epoch + 1} averaged training distance loss: {p_avg_loss:.4f}", flush=True)
            print(f"Pseudo-epoch {pseudo_epoch + 1} averaged training acc: {p_avg_acc:.4f}", flush=True)
            writer.add_scalar("loss/train_p_avg", p_avg_loss, g_step)
            writer.add_scalar("acc/train_p_avg", p_avg_acc, g_step)

            # exponential learning rate decay
            conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((pseudo_epoch + 1) / config.lr_decay_freq)
            dense_lr = dense_lr * config.lr_decay_rate ** ((pseudo_epoch + 1) / config.lr_decay_freq)
            # fmt: off
            optimizer = torch.optim.SGD([
                {'params': model.features.parameters(), 'lr': conv_base_lr},
                {'params': model.classifier.parameters(), 'lr': dense_lr}],
                momentum=0.9)
            # fmt: on

            # do validation after each epoch
            batch_val_losses = []
            batch_val_acc = []
            print("validating")
            for i, data in enumerate(Pexels_val_loader):
                if i * config.val_batch_size > config.val_imgs_count:
                    break
                img1 = data["img1"].to(device)
                img2 = data["img2"].to(device)

                with torch.no_grad():
                    out1, out2 = model(img1, img2, "siamese")

                val_loss = dist_loss(out1, out2, epsilon=config.margin)
                batch_val_losses.append(val_loss.item())

                val_acc = accuracy(out1, out2)
                batch_val_acc.append(val_acc)

            p_avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
            p_avg_val_acc = sum(batch_val_acc) / len(batch_val_acc)

            print(f"Pseudo-epoch {pseudo_epoch + 1} completed. Averaged distance loss on val set: {p_avg_val_loss:.4f}.", flush=True)
            print(f"Pseudo-epoch {pseudo_epoch + 1} completed. Averaged acc on val set: {p_avg_val_acc:.4f}.", flush=True)
            writer.add_scalar("loss/val_p_avg", p_avg_val_loss, g_step)
            writer.add_scalar("acc/val_p_avg", p_avg_val_acc, g_step)

            # Use early stopping to monitor training
            if p_avg_val_loss < init_val_loss:
                init_val_loss = p_avg_val_loss
                # save model weights if val loss decreases
                print("Saving model...")
                Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(Path(config.ckpt_path) / f"epoch{epoch + 1}-p_epoch{pseudo_epoch + 1}.pth"))
                print("Done.\n")
                # reset count
                count = 0
            elif p_avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print(f"Val dist loss has not decreased in {config.early_stopping_patience} pseudo-epochs. Training terminated.")
                    break

    print("Training completed.")
    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--val_imgs_count", type=int, default=500000)
    parser.add_argument("--orig_present", action="store_true")
    parser.add_argument("--parameters", type=str, nargs="+")

    # training parameters
    parser.add_argument("--conv_base_lr", type=float, default=3e-7)
    parser.add_argument("--dense_lr", type=float, default=3e-6)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lr_decay_rate", type=float, default=0.95)
    parser.add_argument("--lr_decay_freq", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_per_p_epoch", type=int, default=1000000)

    # misc
    parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/pexels/cold")
    parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/pexels/cold")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--gpu_ids", type=list, default=None)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--warm_start_epoch", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    config = parser.parse_args()

    main(config)
