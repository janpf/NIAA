import logging
import random
import sys
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image

sys.path[0] = "/workspace"
from IA.utils import filename2path, mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

orig_dir = Path("/scratch/pexels/images")
edited_dir = Path("/scratch/pexels/edited_images")

orig_dir_new = Path("/scratch/pexels/images_small")
edited_dir_new = Path("/scratch/pexels/edited_images_small")


file_list = list(orig_dir.iterdir())
random.shuffle(file_list)

selected = ["pexels-photo-3381646", "pexels-photo-2914334", "pexels-photo-2389091.jpeg", "pexels-photo-358549.jpeg", "pexels-photo-3381646.jpeg", "pexels-photo-3540578.jpeg", "pexels-photo-1717728.jpeg", "pexels-photo-1755287.jpeg"]

file_list = [f for f in file_list if f.name in selected]

for p in file_list:
    new_img_p = orig_dir_new / filename2path(p.name)
    logging.info(f"moving from {p} to {new_img_p}")

    if not new_img_p.exists():
        new_img_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            img: Image.Image = transforms.Resize(336)(Image.open(str(p)))
            img.save(str(new_img_p))
        except:
            logging.error(f"{p}, {new_img_p}")
            continue

    for style_change in mapping["styles_changes"]:
        parameter, change = style_change.split(";")
        change = float(change) if "." in change else int(change)

        old_p: Path = edited_dir / parameter / str(change) / p.name
        new_p = edited_dir_new / parameter / str(change) / filename2path(p.name)
        logging.info(f"moving from {old_p} to {new_p}")

        if not new_p.exists():
            new_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                img: Image.Image = transforms.Resize(336)(Image.open(str(old_p)))
                img.save(str(new_p))
            except:
                logging.error(f"{old_p}, {new_p}")
                break
