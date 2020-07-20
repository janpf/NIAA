import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

sys.path[0] = "/workspace"

from model.datasets import PexelsRedis
from model.NIAA import NIAA


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fmt: off
    Pexels_train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    # fmt: on

    base_model = models.vgg16(pretrained=True)
    model = NIAA(base_model)

    Pexels_train = PexelsRedis(mode="train", transforms=Pexels_train_transform)
    Pexels_train_loader = torch.utils.data.DataLoader(Pexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)

    for data in Pexels_train_loader:
        img1 = data["img1"].to(device)
        img2 = data["img2"].to(device)

        with torch.no_grad():
            out1, out2 = model(img1, img2, "siamese")

        # TODO print out1, out2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/pexels/cold")

    config = parser.parse_args()

    main(config)
