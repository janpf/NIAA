import collections
import math
import random
from shutil import copyfile
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Tuple
import os
import numpy as np
from PIL import Image


def edit_image(img_path: str, change: str, value: float) -> Image:
    with tempfile.NamedTemporaryFile(suffix=".xmp") as edit_file:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as out:
            os.remove(out.name) # because darktable can't overwrite...
            with tempfile.TemporaryDirectory() as darktable_config: # because otherwise darktable can't more than one instance in parallel
                if "lcontrast" == change:
                    copyfile("./darktable_xmp/localcontrast.xmp", edit_file.name) # TODO edit before copying
                    subprocess.run(["darktable-cli", img_path, edit_file.name, out.name, "--core", "--library", "':memory:'", "--configdir", darktable_config])
            return Image.open(out.name)

def edit_image_mp(img_path: str, change: str, value: float, q: multiprocessing.SimpleQueue):
    q.put(edit_image(img_path, change, value))

parameter_range = collections.defaultdict(dict) # TODO redo for darktables
parameter_range["lcontrast"]["min"] = 0
parameter_range["lcontrast"]["default"] = 0  # i think default is impossible (possibly due to c bindings type conversions)
parameter_range["lcontrast"]["max"] = 40

parameter_range["saturation"]["min"] = 0
parameter_range["saturation"]["default"] = 1
parameter_range["saturation"]["max"] = 2

parameter_range["exposure"]["min"] = -10
parameter_range["exposure"]["default"] = 0
parameter_range["exposure"]["max"] = 10

parameter_range["temperature"]["min"] = 1000
parameter_range["temperature"]["default"] = 6500
parameter_range["temperature"]["max"] = 12000

parameter_range["hue"]["min"] = -180
parameter_range["hue"]["default"] = 0
parameter_range["hue"]["max"] = 180

parameter_range["brightness"]["min"] = 0
parameter_range["brightness"]["default"] = 1
parameter_range["brightness"]["max"] = 2

parameter_range["contrast"]["min"] = 0
parameter_range["contrast"]["default"] = 1
parameter_range["contrast"]["max"] = 2

parameter_range["shadows"]["min"] = -100
parameter_range["shadows"]["default"] = 0
parameter_range["shadows"]["max"] = 100

parameter_range["highlights"]["min"] = -100
parameter_range["highlights"]["default"] = 0
parameter_range["highlights"]["max"] = 100


def random_parameters() -> Tuple[str, Tuple[float, float]]: # TODO redo for darktables

    change = random.choice(list(parameter_range.keys()))

    if change == "lcontrast":
        pos_neg = random.choice(["positiv", "interval"])
        lcontrast_vals = [round(val, 1) for val in list(np.arange(0.1, 1, 0.1)) + list(range(1, 10)) + list(range(10, 41, 5))]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        if pos_neg == "positive":
            changeVal = (0, random.choice(lcontrast_vals))
        else:
            changeVal = (random.choice(lcontrast_vals), random.choice(lcontrast_vals))

    elif change == "hue":
        pos_neg = random.choice(["positiv", "negative", "interval"])  # in order to not match a positive change with a negative one

        if pos_neg == "positive":
            changeVal = (0, random.choice(np.arange(1, 11, 1)))
        elif pos_neg == "negative":
            changeVal = (0, random.choice(np.arange(-10, 0, 1)))
        else:
            hue_space = random.choice(list(np.arange(-10, 0, 1)) + list(np.arange(1, 11, 1)))
            changeVal = (hue_space, hue_space)
            while not math.copysign(1, changeVal[0]) == math.copysign(1, changeVal[1]):  # make sure to not compare an image to another one, which has been edited in the other "direction"
                changeVal = (changeVal[0], hue_space)

    else:
        pos_neg = random.choice(["positiv", "negative", "interval"])
        if pos_neg == "positive":
            changeVal = (parameter_range[change]["default"], random.choice(np.linspace(parameter_range[change]["default"], parameter_range[change]["max"], 10)))
        elif pos_neg == "negative":
            changeVal = (parameter_range[change]["default"], random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["default"], 10)))
        else:
            changeVal = (random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["max"], 20)), random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["max"], 20)))
            #fmt:off
            while (changeVal[0] < parameter_range[change]["default"] and changeVal[1] > parameter_range[change]["default"]
                    ) or (
                   changeVal[0] > parameter_range[change]["default"] and changeVal[1] < parameter_range[change]["default"]):
            #fmt:on # make sure to not compare an image to another one, which has been edited in the other "direction
                changeVal = (changeVal[0], random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["max"], 20)))
        changeVal = (round(changeVal[0], 1), round(changeVal[1], 1))
    return change, changeVal

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to a single image", default="/data/442284.jpg")
    parser.add_argument("--parameter", type=str, help="what to change: brightness, contrast...")
    parser.add_argument("--value", type=float, help="change value")
    parser.add_argument("--out", type=str, help="dest for edited images", default="/data/output.jpg")
    args = parser.parse_args()

    # (Path(args.out) / args.parameter).mkdir(parents=True, exist_ok=True)
    # outfile = Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{args.value}.jpg"

    # print(outfile)
    edit_image(img_path=args.image, change=args.parameter, value=args.value).save(args.out)
