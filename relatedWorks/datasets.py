import json
import math
import random
from ctypes import c_wchar_p
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import redis
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from imagenet_c import corrupt
from PIL import Image
import logging

from edit_image import parameter_range


class SSPexels(torch.utils.data.Dataset):
    def __init__(self, file_list_path: str, mapping, normalize: bool = True, orig_dir: str = "/scratch/pexels/images", edited_dir: str = "/scratch/pexels/edited_images"):
        self.file_list_path = file_list_path
        self.mapping = mapping
        self.normalize = normalize

        self.orig_dir = orig_dir
        self.edited_dir = edited_dir

        with open(file_list_path) as f:
            file_list = f.readlines()

        self.file_list = [line.strip() for line in file_list]

        def pixelate(x: Image.Image, severity=1) -> Image.Image:
            previous = x.size
            c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

            x = x.resize((int(previous[0] * c), int(previous[1] * c)), Image.BOX)
            x = x.resize(previous, Image.BOX)

            return x

        self.pixelate = pixelate

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        return self._actualgetitem(idx)

    def _actualgetitem(self, idx):
        data = dict()
        data["original"] = transforms.Resize(256)(Image.open(str(Path(self.orig_dir) / self.file_list[idx])).convert("RGB"))

        for style_change in self.mapping["styles_changes"]:
            parameter, change = style_change.split(";")
            change = float(change) if "." in change else int(change)
            data[style_change] = transforms.Resize(256)(Image.open(str(Path(self.edited_dir) / parameter / str(change) / self.file_list[idx])).convert("RGB"))

        for technical_change in self.mapping["technical_changes"]:
            parameter, change = technical_change.split(";")
            change = int(change)
            if parameter == "pixelate":
                data[technical_change] = self.pixelate(transforms.CenterCrop(224)(data["original"]), severity=change)
            else:
                img = corrupt(np.array(transforms.CenterCrop(224)(data["original"])), severity=change, corruption_name=parameter)
                data[technical_change] = Image.fromarray(img)

        for composition_change in self.mapping["composition_changes"]:
            parameter, change = composition_change.split(";")
            change = int(change)

            if "ratio" == parameter:
                img_size = (data["original"].size[1], data["original"].size[0])
                if change > 0:
                    img_resize = (img_size[0] * (1 + change * (1 / 5)), img_size[1])
                else:
                    img_resize = (img_size[0], img_size[1] * (1 + -change * (1 / 5)))
                img_resize = (round(img_resize[0]), round(img_resize[1]))
                img = transforms.Resize(img_resize)(data["original"])
                data[composition_change] = transforms.CenterCrop(img_size)(img)

            elif "rotate" == parameter:
                rotated = data["original"].rotate(change, Image.BICUBIC, True)

                aspect_ratio = float(data["original"].size[0]) / data["original"].size[1]
                rotated_aspect_ratio = float(rotated.size[0]) / rotated.size[1]
                angle = math.fabs(change) * math.pi / 180

                if aspect_ratio < 1:
                    total_height = float(data["original"].size[0]) / rotated_aspect_ratio
                else:
                    total_height = float(data["original"].size[1])

                h = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
                w = h * aspect_ratio
                data[composition_change] = transforms.CenterCrop((h, w))(rotated)

            elif "crop" in parameter:
                original = data["original"]
                crop_size = transforms.Resize(224)(data["original"]).size

                center_left = round(original.size[0] / 2 - crop_size[0] / 2)
                center_right = round(original.size[0] / 2 + crop_size[0] / 2)
                center_top = round(original.size[1] / 2 - crop_size[1] / 2)
                center_bottom = round(original.size[1] / 2 + crop_size[1] / 2)

                v_move = 0  # centered
                h_move = 0  # centered
                if parameter == "vcrop":
                    v_move = change
                elif parameter == "hcrop":
                    h_move = change
                elif parameter == "leftcornerscrop":
                    h_move = -abs(change)
                    v_move = change
                elif parameter == "rightcornerscrop":
                    h_move = abs(change)
                    v_move = change

                offset_left = round(center_left * (-h_move * (0.2)))
                offset_top = round(center_top * (v_move * (0.2)))

                center_left -= offset_left
                center_right -= offset_left
                center_top -= offset_top
                center_bottom -= offset_top

                data[composition_change] = original.crop((center_left, center_top, center_right, center_bottom))

        for k in data.keys():
            data[k] = transforms.CenterCrop(224)(data[k])
            if self.normalize:
                data[k] = transforms.ToTensor()(data[k])
            else:
                data[k] = np.array(data[k])
                data[k] = np.moveaxis(data[k], -1, 0)
                data[k] = torch.from_numpy(data[k])
                data[k] = data[k].float()
        data["file_name"] = self.file_list[idx]
        return data
