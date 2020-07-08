import json
import math
from pathlib import Path
from typing import Dict, List

import multiprocessing as mp
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from ctypes import c_wchar_p

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

    def __init__(self, file_list_path: str, original_present: bool, available_parameters: List[str], transforms: transforms, orig_dir: str = "/scratch/stud/pfister/NIAA/pexels/images", edited_dir: str = "/scratch/stud/pfister/NIAA/pexels/edited_images"):
        self.file_list_path: str = file_list_path
        with open(file_list_path) as f:
            self.file_list = [val.strip() for val in f.readlines()]
        self.orig_dir: Path = Path(orig_dir)
        self.edited_dir: Path = Path(edited_dir)
        self.original_present: bool = original_present
        self.available_parameters: List[str] = available_parameters  # ["brightness", "contrast", ..]
        self.transforms: transforms = transforms

        length = 0
        for img in self.file_list:
            for parameter in self.available_parameters:
                for change in parameter_range[parameter]["range"]:
                    if self.original_present:
                        if math.isclose(change, parameter_range[parameter]["default"]):
                            continue
                        length += 1
        self.edits = mp.Array(c_wchar_p, length, lock=False)  # lock false, as the list will be effectively readonly

        i = 0
        for img in self.file_list:
            for parameter in self.available_parameters:
                for change in parameter_range[parameter]["range"]:
                    if self.original_present:
                        if math.isclose(change, parameter_range[parameter]["default"]):
                            continue
                        relDist = abs((parameter_range[parameter]["default"]) - (change))
                        relDist = 0 if math.isclose(relDist, 0) else relDist
                        relDist = round(relDist, 2)
                        self.edits[i] = json.dumps({"img1": str(self.orig_dir / img), "img2": str(self.edited_dir / parameter / str(change) / img), "parameter": parameter, "changes1": parameter_range[parameter]["default"], "changes2": change, "relChanges1": 0, "relChanges2": relDist})
                        i += 1
                    else:
                        raise NotImplementedError("bruh")

    def __len__(self) -> int:
        return len(self.edits)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = json.loads(self.edits[idx])

        item["img1"] = self.transforms(Image.open(item["img1"]).convert("RGB"))
        item["img2"] = self.transforms(Image.open(item["img2"]).convert("RGB"))

        return item
