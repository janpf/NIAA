import json
import math
import random
from ctypes import c_wchar_p
from pathlib import Path
from typing import Dict, List

import pandas as pd
import redis
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from PIL import Image

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
        row = self.annotations.iloc[idx]  #  TODO wie funktioniert iloc?

        img = Image.open(self.root_dir / (row["img_id"] + ".jpg")).convert("RGB")  # FIXME, but not yet
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
        return self.transforms(Image.open(self.file_list[idx]).convert("RGB"))


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


class PexelsDistortRedis(torch.utils.data.Dataset):
    """Pexels dataset

    Args:
        mode: train, val or test
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, mode: str, transforms: transforms, distortion_fn, **dist_kwargs):
        self.db_host = "redisdataset"
        self.mode = mode
        self.transforms = transforms

        self.distortion_fn = distortion_fn
        self.dist_kwargs = dist_kwargs

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
            item["img1"] = self.transforms(self.distortion_fn(item["img1"], self.dist_kwargs))
            item["img2"] = self.transforms(self.distortion_fn(item["img2"], self.dist_kwargs))
            return item
        except:
            return self[random.randint(0, len(self))]  # if an image is broken
