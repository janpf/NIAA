from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


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

    def __getitem__(self, idx) -> AVASample:
        row = self.annotations.iloc[idx]  #  TODO wie funktioniert iloc?

        img = Image.open(self.root_dir / (row["img_id"] + ".jpg")).convert("RGB")  # TODO kontrollieren
        img = self.transforms(img)

        distribution = list(row.loc[:, "1":"10"])
        distribution = torch.FloatTensor(distribution)
        distribution = torch.nn.Softmax(dim=-1)(distribution)
        score = distribution.dot(torch.FloatTensor(list(range(1, len(distribution) + 1))))

        return {"img_id": row["img_id"], "img": img, "distribution": distribution, "score": score}


class Pexels(torch.utils.data.Dataset):
    """Pexels dataset

    Args:
        csv_file: a csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        edited_dir: directory to the edited images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file: str, root_dir: str, edited_dir: str, transforms: transforms):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file, header=["img1", "img2", "parameter", "changes1", "changes2", "relChanges1", "relChanges2"])
        self.root_dir: Path = Path(root_dir)
        self.edited_dir: Path = Path(edited_dir)
        self.transforms: transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> PexelsSample:
        row = self.annotations.iloc[idx]

        img1 = Image.open(self.root_dir / row["img1"]).convert("RGB")
        img1 = self.transforms(img1)

        img2 = Image.open(self.edited_dir / row["img2"] + ".jpg").convert("RGB")
        img2 = self.transforms(img2)

        return {"img1": img1, "img2": img2, "parameter": row["parameter"], "changes1": row["changes1"], "changes2": row["changes2"], "relChanges1": row["relChanges1"], "relChanges2": row["relChanges2"]}
