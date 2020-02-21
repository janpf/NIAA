import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pillow_lut as lut
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="path to a single image", default="/scratch/stud/pfister/NIAA/442284.jpg")
parser.add_argument("--parameter", type=str, help="what to change: brightness, contrast...")
parser.add_argument("--value", type=float, help="change value")
parser.add_argument("--out", type=str, help="dest for edited images", default="/scratch/stud/pfister/NIAA/AVA/changed")
args = parser.parse_args()


# brightness – One value for all channels, or tuple of three values from -1.0 to 1.0. Use exposure for better result.
# exposure – One value for all channels, or tuple of three values from -5.0 to 5.0.
# contrast – One value for all channels, or tuple of three values from -1.0 to 5.0.
# warmth – One value from -1.0 to 1.0.
# saturation – One value for all channels, or tuple of three values from -1.0 to 5.0.
# vibrance – One value for all channels, or tuple of three values from -1.0 to 5.0.
# hue – One value from 0 to 1.0.
# l_contrast


def edit_image(img_path: str, changes: Dict[str, float], outpath: str):
    if "lcontrast" in changes:  # CLAHE
        img = cv2.imread(str(img_path))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        cl = cv2.createCLAHE(clipLimit=changes["lcontrast"], tileGridSize=(8, 8)).apply(l)

        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        _, buffer = cv2.imencode(".jpeg", img)
        buffer.tofile(outpath)  # cv2 can't write to a file called [...].0.2.jpg as it doesn't know the extension
    elif "laplace" in changes:  # Local Laplacian Filter
        img = cv2.imread(str(img_path))

        dst = cv2.Laplacian(img, cv2.CV_16S, ksize=1)
        abs_img = cv2.convertScaleAbs(dst)

        _, buffer = cv2.imencode(".jpeg", abs_img)
        buffer.tofile(outpath)  # cv2 can't write to a file called [...].0.2.jpg as it doesn't know the extension
    else:
        img = Image.open(img_path)
        if "lcontrast" in changes:
            del changes["lcontrast"]
        img.convert("RGB").filter(lut.rgb_color_enhance(16, **changes)).save(outpath, "JPEG")


(Path(args.out) / args.parameter).mkdir(parents=True, exist_ok=True)
outfile = Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{args.value}.jpg"

edit_image(img_path=args.image, changes={args.parameter: args.value}, outpath=outfile)
