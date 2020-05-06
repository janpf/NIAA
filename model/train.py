import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import torchvision.transforms as transforms

from datasets import AVA, Pexels
from model import NIAA, Distance_Loss, Earth_Movers_Distance_Loss


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fmt: off
    AVA_train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    AVA_val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    Pexels_train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    Pexels_val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    # fmt: on
    model = NIAA()

    if config.warm_start:
        model.load_state_dict(torch.load(str(Path(config.ckpt_path) / f"epoch-{config.warm_start_epoch}.pkl")))
        print(f"Successfully loaded model epoch-{config.warm_start_epoch}.pkl")

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    # fmt: off
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': config.conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': config.dense_lr}],
        momentum=0.9)
    # fmt: on

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print(f"Trainable params: {(param_num / 1e6):.2f} million")

    if config.train:
        AVA_trainset = AVA(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=AVA_train_transform)
        AVA_valset = AVA(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=AVA_val_transform)

        AVA_train_loader = torch.utils.data.DataLoader(AVA_trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
        AVA_val_loader = torch.utils.data.DataLoader(AVA_valset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers)

        count = 0  # for early stopping
        init_val_loss = float("inf")
        train_losses = []
        val_losses = []
        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            for i, data in enumerate(AVA_train_loader):
                images = data.img.to(device)
                labels = data.distribution.to(device).float()
                outputs = model(images)
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                loss = Earth_Movers_Distance_Loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()
                print(f"Epoch: {epoch + 1}/{config.epochs} | Step: {i + 1}/{len(trainset) // config.train_batch_size + 1} | Training EMD loss: {loss.data[0]:.4f}")

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} averaged training EMD loss: {avg_loss:.4f}")

            # exponential learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                # fmt: off
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9)
                # fmt: on

            # do validation after each epoch
            batch_val_losses = []
            for data in AVA_val_loader:
                images = data.img.to(device)
                labels = data.distribution.to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                val_loss = Earth_Movers_Distance_Loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1} completed. Averaged EMD loss on val set: {avg_val_loss:.4f}." % (epoch + 1, avg_val_loss))

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print("Saving model...")
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
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
    parser.add_argument("--train_img_path", type=str, default="/home/yunxiao/ava_data/train")
    parser.add_argument("--val_img_path", type=str, default="/home/yunxiao/ava_data/val")
    parser.add_argument("--train_csv_file", type=str, default="../train_labels.csv")
    parser.add_argument("--val_csv_file", type=str, default="../val_labels.csv")

    # training parameters
    parser.add_argument("--conv_base_lr", type=float, default=3e-7)
    parser.add_argument("--dense_lr", type=float, default=3e-6)
    parser.add_argument("--lr_decay_rate", type=float, default=0.95)
    parser.add_argument("--lr_decay_freq", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)

    # misc
    parser.add_argument("--ckpt_path", type=str, default="../ckpts")
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--gpu_ids", type=list, default=None)
    parser.add_argument("--warm_start", type=bool, default=False)
    parser.add_argument("--warm_start_epoch", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--save_fig", type=bool, default=False)

    config = parser.parse_args()

    main(config)
