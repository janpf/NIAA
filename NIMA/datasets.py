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


class AVA(torch.utils.data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file: str, root_dir: str, transforms: transforms):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file, delimiter=" ", header=["index", "img_id", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "sem_tag_1", "sem_tag_2", "chall_id"])
        self.root_dir: Path = Path(root_dir)
        self.transforms: transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> Dict[str, str]:
        row = self.annotations.iloc[idx]

        img = Image.open(self.root_dir / (row["img_id"] + ".jpg")).convert("RGB")
        img = self.transforms(img)

        distribution = list(row.loc[:, "1":"10"])
        distribution = torch.FloatTensor(distribution)
        distribution = torch.nn.Softmax(dim=-1)(distribution)
        score = distribution.dot(torch.FloatTensor(list(range(1, len(distribution) + 1))))

        return {"img_id": row["img_id"], "img": img, "distribution": distribution, "score": score}


class Pexels(torch.utils.data.Dataset):
    """Pexels dataset

    Args:
        file_list_path: a file with a list of files to be loaded
        original_present: compare against the regular image
        available_parameters: which parameters to edit
        transform: preprocessing and augmentation of the training images
        orig_dir: directory to the original images
        edited_dir: directory to the edited images
    """

    def __init__(self, file_list_path: str, original_present: bool, compare_opposite_polarity: bool, available_parameters: List[str], transforms: transforms, orig_dir: str = "/scratch/stud/pfister/NIAA/pexels/images", edited_dir: str = "/scratch/stud/pfister/NIAA/pexels/edited_images"):
        raise DeprecationWarning("has been replaced with PexelsRedis below")
        print("initializing dataset", flush=True)
        self.file_list_path: str = file_list_path
        with open(file_list_path) as f:
            self.file_list = [val.strip() for val in f.readlines()]
        self.orig_dir: Path = Path(orig_dir)
        self.edited_dir: Path = Path(edited_dir)
        self.original_present: bool = original_present
        self.compare_opposite_polarity: bool = compare_opposite_polarity
        self.available_parameters: List[str] = available_parameters  # ["brightness", "contrast", ..]
        self.transforms: transforms = transforms

        edits = []

        for img in self.file_list:
            for parameter in self.available_parameters:
                if self.original_present:
                    for change in parameter_range[parameter]["range"]:
                        if math.isclose(change, parameter_range[parameter]["default"]):
                            continue
                        relDist = abs((parameter_range[parameter]["default"]) - (change))
                        relDist = 0 if math.isclose(relDist, 0) else relDist
                        relDist = round(relDist, 2)

                        edits.append(json.dumps({"img1": str(self.orig_dir / img), "img2": str(self.edited_dir / parameter / str(change) / img), "parameter": parameter, "changes1": parameter_range[parameter]["default"], "changes2": change, "relChanges1": 0, "relChanges2": relDist}))

                else:
                    for lchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
                        for rchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
                            if math.isclose(lchange, rchange):  # don't compare 0.5 to 0.5 for example
                                continue
                            if rchange < lchange:  # only compare 0.4 to 0.5 but not 0.5 to 0.4
                                continue
                            if not self.compare_opposite_polarity:
                                if lchange < parameter_range[parameter]["default"] and rchange > parameter_range[parameter]["default"] or lchange > parameter_range[parameter]["default"] and rchange < parameter_range[parameter]["default"]:
                                    continue

                            lRelDist = abs((parameter_range[parameter]["default"]) - (lchange))
                            rRelDist = abs((parameter_range[parameter]["default"]) - (rchange))
                            lRelDist = round(lRelDist, 2)
                            rRelDist = round(rRelDist, 2)

                            if lRelDist > rRelDist:  # smaller change always has to be img1/imgl, as the Distance Loss assumes the "more original" image is the first
                                continue

                            if math.isclose(lchange, parameter_range[parameter]["default"]):
                                imgl = str(self.orig_dir / img)
                            else:
                                imgl = str(self.edited_dir / parameter / str(lchange) / img)
                            if math.isclose(rchange, parameter_range[parameter]["default"]):
                                imgr = str(self.orig_dir / img)
                            else:
                                imgr = str(self.edited_dir / parameter / str(rchange) / img)

                            edits.append(json.dumps({"img1": imgl, "img2": imgr, "parameter": parameter, "changes1": lchange, "changes2": rchange, "relChanges1": lRelDist, "relChanges2": rRelDist}))

        print(f"created all {len(edits)} datapoints", flush=True)
        print("moving datapoints to /dev/shm", flush=True)
        if False:  # if enough memory as peak memory is quite a bit
            self.edits = mp.Array(c_wchar_p, edits, lock=False)  # lock false, as the list will be effectively readonly after insert
        else:
            self.edits = mp.Array(c_wchar_p, len(edits), lock=False)  # lock false, as the list will be effectively readonly after insert
            for i in range(len(edits)):
                self.edits[i] = edits.pop(0)

        del edits
        print("moved datapoints to /dev/shm", flush=True)

    def __len__(self) -> int:
        return len(self.edits)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        try:
            item = json.loads(self.edits[idx])
            item["img1"] = self.transforms(Image.open(item["img1"]).convert("RGB"))
            item["img2"] = self.transforms(Image.open(item["img2"]).convert("RGB"))
            return item
        except:
            return self[random.randint(0, len(self))]  # if an image is broken


class FileList(torch.utils.data.Dataset):
    """FileList dataset

    Args:
        file_list: a list with filenames
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, file_list: List[str], transforms: transforms):
        self.file_list: List[str] = file_list
        self.transforms: transforms = transforms

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {"img": self.transforms(Image.open(self.file_list[idx]).convert("RGB")), "path": self.file_list[idx]}


class FileListDistorted(torch.utils.data.Dataset):
    """FileList dataset

    Args:
        file_list: a list with filenames
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, file_list: List[str]):
        self.file_list: List[str] = file_list
        self.transforms: transforms = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224)])
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        img: Image.Image = self.transforms(Image.open(self.file_list[idx]).convert("RGB"))
        arr = np.array(img)
        data = {"path": self.file_list[idx]}
        for i, corr in enumerate(["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter"]):
            data[f"corr_{i}"] = corr
            for s in range(1, 6):
                corr_arr = corrupt(arr, severity=s, corruption_name=corr)
                data[f"img{i}-{s}"] = self.to_tensor(Image.fromarray(corr_arr.astype("uint8"), "RGB"))
        data["num_corrs"] = i
        return data


class PexelsRedis(torch.utils.data.Dataset):
    """Pexels dataset

    Args:
        mode: train, val or test
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, mode: str, transforms: transforms):
        self.db_host = "redisdataset"
        self.mode = mode
        self.transforms = transforms
        if self.mode == "train":
            self.db = 0
        elif self.mode == "val":
            self.db = 1
        elif self.mode == "test":
            self.db = 2
        else:
            raise NotImplementedError("?")

        print(f"connecting to {mode} db ({self.db})")
        self.db = redis.Redis(host=self.db_host, db=self.db)
        self.size = self.db.dbsize()
        print(f"{self.size} datapoints in db")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.db.get(idx)
        item = json.loads(item)
        try:
            item["img1"] = self.transforms(Image.open(item["img1"]).convert("RGB"))
            item["img2"] = self.transforms(Image.open(item["img2"]).convert("RGB"))
            return item
        except:
            return self[random.randint(0, len(self))]  # if an image is broken


class SSPexels(torch.utils.data.Dataset):
    def __init__(self, file_list_path: str, mapping, orig_dir: str = "/scratch/pexels/images", edited_dir: str = "/scratch/pexels/edited_images"):
        self.file_list_path = file_list_path
        self.mapping = mapping

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
        try:
            logging.info(self.file_list[idx])
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx):
        data = dict()
        data["original"] = transforms.Resize(256)(Image.open(str(Path(self.orig_dir) / self.file_list[idx])).convert("RGB"))

        for style_change in self.mapping["style_changes"]:
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
            data[k] = transforms.ToTensor()(data[k])

        data["file_name"] = self.file_list[idx]
        return data
