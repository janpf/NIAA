import argparse
from pathlib import Path

import numpy as np

from skimage import img_as_float, exposure, io
from PIL import Image
import pillow_lut as lut

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="path to a single image")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images (if this is supplied '--image' is ignored)")
parser.add_argument("--parameter", type=str, help="what to change: brightness, contrast...")
parser.add_argument("--min_range", type=float, help="range to change the parameter in", default=-1)
parser.add_argument("--max_range", type=float, help="range to change the parameter in", default=1)
parser.add_argument("--step", type=float, help="step size of changes", default=0.2)
parser.add_argument("--out", type=str, help="dest for edited images")
args = parser.parse_args()


# brightness
# exposure
# contrast
# warmth
# saturation
# vibrance
# hue
# l_contrast


def edit_image(img_path):
    if args.parameter == "l_contrast":
        img = img_as_float(img_path)
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        io.imsave(Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{args.change}_.jpg", img_adapteq)

    else:
        for change in np.arange(args.min_range, args.max_range + 0.001, args.step):
            img = Image.open(img_path)
            img_filter = lut.rgb_color_enhance(16, **{args.parameter: change})  # does this work?
            img.filter(img_filter).save(Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{args.change}_.jpg")


if args.imageFolder:
    for img in Path(args.imageFolder).iterdir():
        edit_image(str(img))
elif args.image:
    edit_image(args.image)
else:
    raise ("What do you expect me to do?")
