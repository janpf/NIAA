import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from train_pre.utils import filename2path
from train_pre import preprocess_images


class SSPexelsSmall(torch.utils.data.Dataset):
    def __init__(
        self,
        file_list_path: str,
        mapping,
        return_file_name: bool = False,
        normalize: bool = True,
        orig_dir: str = "/scratch/pexels/images_small",
    ):
        self.file_list_path = file_list_path
        self.mapping = mapping

        self.return_file_name = return_file_name
        self.normalize = normalize
        self.orig_dir = orig_dir

        with open(file_list_path) as f:
            file_list = f.readlines()

        file_list = [line.strip() for line in file_list]
        self.file_list = [filename2path(p) for p in file_list]

        self.ed = preprocess_images.ImageEditor()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        # logging.debug(f"getting datapoint {idx}")
        # return self._actualgetitem(idx)
        try:
            logging.debug(f"getting datapoint {idx}")
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx):
        data = self.ed.distort_list_image(
            path=str(Path(self.orig_dir) / self.file_list[idx]),
            distortion_intens_tuple_list=self.mapping["all_changes"],
        )

        for k in data.keys():
            data[k] = transforms.Resize(224)(data[k])
            data[k] = data[k].convert("RGB")
            data[k] = self.ed.pad_square(data[k])
            data[k] = transforms.ToTensor()(data[k])
            if self.normalize:
                data[k] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data[k])

        if self.return_file_name:
            data["file_name"] = self.file_list[idx]
        return data


class SSPexelsDummy(torch.utils.data.Dataset):
    def __init__(self, file_list_path: str, mapping, return_file_name: bool = False):
        self.file_list_path = file_list_path
        self.mapping = mapping

        with open(file_list_path) as f:
            file_list = f.readlines()

        self.file_list = [line.strip() for line in file_list]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        items = ["original", "crop_original", "rotate_original"]
        for style_change in self.mapping["styles_changes"]:
            items.append(style_change)

        for technical_change in self.mapping["technical_changes"]:
            items.append(technical_change)

        for composition_change in self.mapping["composition_changes"]:
            items.append(composition_change)

        data = dict()
        for item in items:
            data[item] = torch.rand(3, 224, 224)
            data[item] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data[item])

        return data


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, normalize: bool, accepted_extensions: List[str] = ["jpg", "bmp", "png"]):
        self.image_dir = image_dir
        self.normalize = normalize
        self.files = [
            str(val) for val in Path(image_dir).glob("**/*") if val.name.split(".")[-1].lower() in accepted_extensions
        ]
        logging.info(f"found {len(self.files)} files")

        def pad_square(im: Image.Image, min_size: int = 224, fill_color=(0, 0, 0)) -> Image.Image:
            im = transforms.Resize(224)(im)
            x, y = im.size
            size = max(min_size, x, y)
            new_im = Image.new("RGB", (size, size), fill_color)
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return transforms.Resize(224)(new_im)

        self.pad_square = pad_square

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.pad_square(img)
        img = transforms.ToTensor()(img)
        if self.normalize:
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return {"img": img, "path": path, "idx": idx}


class AVA(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/AVA/images", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)


class TID2013(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/tid2013", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)

    def __getitem__(self, idx: int):
        return self._actualgetitem(idx)


class KADID10k(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/kadid10k/images", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)

    def __getitem__(self, idx: int):
        return self._actualgetitem(idx)


class Unsplash(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/unsplash/images", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)

    def __getitem__(self, idx: int):
        return self._actualgetitem(idx)


class FIVEK(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/fivek", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize, accepted_extensions=["tif"])
