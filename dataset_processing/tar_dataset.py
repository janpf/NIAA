import logging
import random
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage

sys.path[0] = "."

from SSMTIA.utils import filename2path, mapping


class SSPexels(Dataset):
    def __init__(self, file_list_path: str, mapping, return_file_name: bool = False, orig_dir: str = "/scratch/pexels/images", edited_dir: str = "/scratch/pexels/edited_images"):
        self.file_list_path = file_list_path
        self.mapping = mapping

        self.return_file_name = return_file_name
        self.orig_dir = orig_dir
        self.edited_dir = edited_dir

        self.img_size = 256

        with open(file_list_path) as f:  # TODO all
            file_list = f.readlines()

        self.file_list = [line.strip() for line in file_list]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            print("broken:", idx, self.file_list[idx])
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx):
        data = dict()
        data["original"] = transforms.Resize(self.img_size)(Image.open(str(Path(self.orig_dir) / self.file_list[idx])).convert("RGB"))

        for style_change in self.mapping["style_changes"]:
            parameter, change = style_change.split(";")
            change = float(change) if "." in change else int(change)
            data[style_change] = transforms.Resize(self.img_size)(Image.open(str(Path(self.edited_dir) / parameter / str(change) / self.file_list[idx])).convert("RGB"))

        for key in data.keys():
            data[key] = transforms.ToTensor()(data[key])

        if self.return_file_name:
            data["file_name"] = self.file_list[idx]

        return data


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

file_list_file = "/workspace/dataset_processing/train_set.txt"
out_tar = "/scratch/pexels/images.tar"


with open(file_list_file) as f:
    file_list = [line.strip() for line in f.readlines()]

with open("/workspace/dataset_processing/val_set.txt") as f:
    file_list.extend([line.strip() for line in f.readlines()])

with open("/workspace/dataset_processing/test_set.txt") as f:
    file_list.extend([line.strip() for line in f.readlines()])

dataset = SSPexels(file_list_path=file_list_file, mapping=mapping, return_file_name=True)
dataload = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=50)

out_tar = tarfile.open(out_tar, "w:gz", bufsize=1024 * 1000)

already_done = set()

toPIL = transforms.ToPILImage()
for data in dataload:
    file_name = data["file_name"][0]
    file_name = Path(file_name).stem + ".jpg"

    if file_name in already_done:
        continue
    else:
        already_done.add(file_name)

    logging.info(file_name)
    for key in data.keys():
        if key == "file_name":
            continue

        if key != "original":
            parameter, change = key.split(";")
            change = float(change) if "." in change else int(change)

            path_in_tar = str(Path(parameter) / str(change) / filename2path(file_name))
        else:
            parameter = "original"
            path_in_tar = str(Path(parameter) / filename2path(file_name))

        with BytesIO() as buffer:
            toPIL(data[key][0]).save(buffer, format="JPEG")
            info = tarfile.TarInfo(name=path_in_tar)
            info.size = buffer.tell()
            buffer.seek(0)
            out_tar.addfile(tarinfo=info, fileobj=buffer)

    logging.info("waiting for new file")
