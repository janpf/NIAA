import argparse
from pathlib import Path

import numpy as np

from skimage import exposure, io
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


# brightness – One value for all channels, or tuple of three values from -1.0 to 1.0. Use exposure for better result.
# exposure – One value for all channels, or tuple of three values from -5.0 to 5.0.
# contrast – One value for all channels, or tuple of three values from -1.0 to 5.0.
# warmth – One value from -1.0 to 1.0.
# saturation – One value for all channels, or tuple of three values from -1.0 to 5.0.
# vibrance – One value for all channels, or tuple of three values from -1.0 to 5.0.
# hue – One value from 0 to 1.0.
# l_contrast


def edit_image(img_path):
    if args.parameter == "lcontrast":
        img = io.imread(img_path)
        clip_limit = 0.03
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        io.imsave(Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{clip_limit:.3f}_.jpg", img_adapteq)  # Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.

    else:
        img = Image.open(img_path)
        for change in np.arange(args.min_range, args.max_range + (args.step / 2), args.step):
            img_filter = lut.rgb_color_enhance(16, **{args.parameter: change})
            img.filter(img_filter).save(Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{change:.1f}_.jpg")


(Path(args.out) / args.parameter).mkdir(parents=True, exist_ok=True)
if args.imageFolder:
    for img in Path(args.imageFolder).iterdir():
        edit_image(str(img))
elif args.image:
    edit_image(args.image)
else:
    raise ("What do you expect me to do?")
