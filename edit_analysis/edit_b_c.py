import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="path to a single image")
parser.add_argument("--out", type=str, help="dest for edited images")
args = parser.parse_args()

im = Image.open(args.image)
im.save(Path(args.out) / "orig.png")

def change_brightness(im, bright):
    enhancer = ImageEnhance.Brightness(im)
    enhanced_im = enhancer.enhance(bright)
    return enhanced_im

def change_contrast(im, contra):
    enhancer = ImageEnhance.Contrast(im)
    enhanced_im = enhancer.enhance(contra)
    return enhanced_im

for b in np.arange(0, 3.1, .2):
    for c in np.arange(0, 3.1, .2):
        print(f"creating b={b:.1f} and c={c:.1f}")
        changed = change_contrast(im, c)
        changed = change_brightness(changed, b)
        changed.save(Path(args.out) / f"b{b:.1f}_c{c:.1f}.png")
