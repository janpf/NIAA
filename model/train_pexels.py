import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms

sys.path[0] = "/workspace"

from model.datasets import Pexels
from model.NIAA import NIAA, Distance_Loss


def main(config):
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
    model = NIAA(base_model_pretrained=True)

    if config.warm_start:
        Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
        model.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch-{config.warm_start_epoch}.pkl")))
        print(f"Successfully loaded model epoch-{config.warm_start_epoch}.pkl")
    else:
        print("starting cold")

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    # fmt: off
    optimizer = torch.optim.SGD([
        {'params': model.features.parameters(), 'lr': config.conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': config.dense_lr}],
        momentum=0.9)
    # fmt: on

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print(f"Trainable params: {(param_num / 1e6):.2f} million")

    Pexels_train = Pexels(file_list_path=config.train_files, original_present=config.orig_present, available_parameters=config.parameters, transforms=Pexels_train_transform, orig_dir=config.original_img_dir, edited_dir=config.edited_img_dir)
    Pexels_val = Pexels(file_list_path=config.val_files, original_present=config.orig_present, available_parameters=config.parameters, transforms=Pexels_val_transform, orig_dir=config.original_img_dir, edited_dir=config.edited_img_dir)

    Pexels_train_loader = torch.utils.data.DataLoader(Pexels_train, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    Pexels_val_loader = torch.utils.data.DataLoader(Pexels_val, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers)

    count = 0  # for early stopping
    init_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(config.warm_start_epoch, config.epochs):
        batch_losses = []
        for i, data in enumerate(Pexels_train_loader):
            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            out1, out2 = model(img1, img2, mode="siamese")

            optimizer.zero_grad()

            loss = Distance_Loss(out1, out2)
            batch_losses.append(loss.item())

            loss.backward()

            optimizer.step()
            print(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(Pexels_train_loader)} | Training EMD loss: {loss.data[0]:.4f}")

        avg_loss = sum(batch_losses) / len(Pexels_train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} averaged training EMD loss: {avg_loss:.4f}")

        # exponential learning rate decay
        if (epoch + 1) % 10 == 0:
            conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
            dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
            # fmt: off
            optimizer = torch.optim.SGD([
                {'params': model.features.parameters(), 'lr': conv_base_lr},
                {'params': model.classifier.parameters(), 'lr': dense_lr}],
                momentum=0.9)
            # fmt: on

        # do validation after each epoch
        batch_val_losses = []
        for data in Pexels_val_loader:
            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)

            with torch.no_grad():
                out1, out2 = model(img1, img2, mode="siamese")

            val_loss = Distance_Loss(out1, out2)
            batch_val_losses.append(val_loss.item())
        avg_val_loss = sum(batch_val_losses) / len(Pexels_val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1} completed. Averaged EMD loss on val set: {avg_val_loss:.4f}." % (epoch + 1, avg_val_loss))

        # Use early stopping to monitor training
        if avg_val_loss < init_val_loss:
            init_val_loss = avg_val_loss
            # save model weights if val loss decreases
            print("Saving model...")
            Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(Path(config.ckpt_path) / f"epoch-{epoch + 1}.pkl"))
            print("Done.\n")
            # reset count
            count = 0
        elif avg_val_loss >= init_val_loss:
            count += 1
            if count == config.early_stopping_patience:
                print(f"Val EMD loss has not decreased in {config.early_stopping_patience} epochs. Training terminated.")
                break

    print("Training completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--original_img_dir", type=str, default="/scratch/pexels/images")
    parser.add_argument("--edited_img_dir", type=str, default="/scratch/pexels/edited_images")
    parser.add_argument("--train_files", type=str, default="/workspace/dataset_processing/train_set.txt")
    parser.add_argument("--val_files", type=str, default="/workspace/dataset_processing/val_set.txt")
    parser.add_argument("--orig_present", action="store_true")
    parser.add_argument("--parameters", type=str, nargs="+")

    # training parameters
    parser.add_argument("--conv_base_lr", type=float, default=3e-7)
    parser.add_argument("--dense_lr", type=float, default=3e-6)
    parser.add_argument("--lr_decay_rate", type=float, default=0.95)
    parser.add_argument("--lr_decay_freq", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)

    # misc
    parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/pexels/cold")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--gpu_ids", type=list, default=None)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--warm_start_epoch", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    config = parser.parse_args()

    main(config)
