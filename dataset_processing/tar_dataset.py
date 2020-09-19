import logging
import random
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path[0] = "."

from SSMTIA.utils import filename2path, mapping


class SSPexels(Dataset):
    def __init__(self, file_list: str, mapping, return_file_name: bool = False, orig_dir: str = "/scratch/pexels/images", edited_dir: str = "/scratch/pexels/edited_images"):
        self.mapping = mapping

        self.return_file_name = return_file_name
        self.orig_dir = orig_dir
        self.edited_dir = edited_dir

        self.img_size = 256

        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            logging.info(f"broken:\t{idx}\t{self.file_list[idx]}")
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
out_tar = "/scratch/pexels/images.tar.gz"
patch_tar = "/scratch/pexels/images-patch2.tar"

with open(file_list_file) as f:
    file_list = [line.strip() for line in f.readlines()]

with open("/workspace/dataset_processing/val_set.txt") as f:
    file_list.extend([line.strip() for line in f.readlines()])

with open("/workspace/dataset_processing/test_set.txt") as f:
    file_list.extend([line.strip() for line in f.readlines()])

logging.info(f"files: {len(file_list)}")
logging.info(f"files unique: {len(set(file_list))}")

already_done = set()
if False:
    out_tar = tarfile.open(out_tar, "r", bufsize=1024 * 1000)

    for i, f in enumerate(out_tar):
        if i % 1000000 == 0:
            logging.info(f"done: {i}")
        already_done.add(Path(f.name).stem + ".jpg")

    out_tar.close()
else:
    with open("/workspace/analysis/not_uploaded/done2") as f:
        already_done.update(eval([line.strip() for line in f.readlines()][0]))

missing = set([Path(val).stem + ".jpg" for val in file_list]).difference(already_done)
logging.info(f"missing: {len(missing)}")

file_list = [val for val in file_list if Path(val).stem + ".jpg" in missing]
logging.info(f"creating now: {len(file_list)}")

dataset = SSPexels(file_list=file_list, mapping=mapping, return_file_name=True)
dataload = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=50)

patch_tar = tarfile.open(patch_tar, "w", bufsize=1024 * 1000)

logging.info("appending new files")
toPIL = transforms.ToPILImage()
for i, data in enumerate(dataload):
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
            patch_tar.addfile(tarinfo=info, fileobj=buffer)

    if i % 100 == 0:
        logging.info(f"already done: {len(already_done)}")


print(already_done)
