import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class AVA(Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transforms: transforms):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file, delimiter=" ", header=["index", "img_id", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "sem_tag_1", "sem_tag_2", "chall_id"])
        self.root_dir: Path = Path(root_dir)
        self.transforms: transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> dict:
        row = self.annotations.iloc[idx]  #  TODO wie funktioniert iloc?
        img = Image.open(self.root_dir / (row["img_id"] + ".jpg"))  # TODO kontrollieren
        img = self.transforms(img)

        sample = {"img_id": row["img_id"], "image": img, "annotations": list(row.loc[:, "1":"10"])}

        return sample


# TODO pexels dataset
