import collections
import math
import random
import subprocess
import tempfile
from pathlib import Path
from struct import pack, unpack
from typing import Dict, Tuple

import cv2
import numpy as np
from jinja2 import Template
from PIL import Image


def edit_image(img_path: str, change: str, value: float) -> Image:

    if "lcontrast" == change:  # XXX localcontrast xmp in darktable is broken atm. no idea why
        img = cv2.imread(img_path)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        cl = cv2.createCLAHE(clipLimit=value, tileGridSize=(8, 8)).apply(l)

        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

    with tempfile.TemporaryDirectory() as darktable_config:  # because otherwise darktable can't open more than one instance in parallel
        edit_file = str(Path(darktable_config) / "edit.xmp")
        out_file = str(Path(darktable_config) / "out.jpg")

        if "contrast" == change or "brightness" == change or "saturation" == change:
            template_file = "./darktable_xmp/colisa.xmp"
            param_index = ["contrast", "brightness", "saturation"].index(change)
            default_str = "".join(["%02x" % b for b in bytearray(pack("f", 0))])
            change_val_enc = "".join(["%02x" % b for b in bytearray(pack("f", value))])
            change_str = "".join([change_val_enc if _ == param_index else default_str for _ in range(3)])

        if "shadows" == change or "highlights" == change:
            template_file = "./darktable_xmp/shadhi.xmp"
            if "shadows" == change:
                change_str = f"000000000000c842{''.join(['%02x' % b for b in bytearray(pack('f', value))])}000000000000c84200000000000048420000c842000048427f000000bd37863500000000"
            elif "highlights" == change:
                change_str = f"000000000000c8420000484200000000{''.join(['%02x' % b for b in bytearray(pack('f', value))])}00000000000048420000c842000048427f000000bd37863500000000"

        if "exposure" == change:
            template_file = "./darktable_xmp/exposure.xmp"
            change_str = f"0000000000000000{''.join(['%02x' % b for b in bytearray(pack('f', value))])}00004842000080c0"  # TODO check

        if "vibrance" == change:
            template_file = "./darktable_xmp/vibrance.xmp"
            change_str = "".join(["%02x" % b for b in bytearray(pack("f", value))])

        if "temperature" == change or "tint" == change:
            raise ("aaaarg")
            template_file = "./darktable_xmp/temperature.xmp"
            if "temperature" == change:
                change_str = f"f3efbf3f0000803fa91a073f0000807f"  # TODO check
            elif "tint" == change:
                change_str = f"f3efbf3f0000803fa91a073f0000807f"  # TODO check

        with open(template_file) as template_file:
            Template(template_file.read()).stream(value=change_str).dump(edit_file)

        subprocess.run(["darktable-cli", img_path, edit_file, out_file, "--core", "--library", ":memory:", "--configdir", darktable_config])
        return Image.open(out_file)


def edit_image_mp(img_path: str, change: str, value: float, q):
    q.put(edit_image(img_path, change, value))


parameter_range = collections.defaultdict(dict)  # TODO redo for darktables
parameter_range["contrast"]["min"] = -1
parameter_range["contrast"]["default"] = 0
parameter_range["contrast"]["max"] = 1

parameter_range["brightness"] = parameter_range["contrast"]
parameter_range["saturation"] = parameter_range["contrast"]

parameter_range["shadows"]["min"] = -100  # wahrscheinlich 3. Packen
parameter_range["shadows"]["max"] = 100

parameter_range["highlights"] = parameter_range["shadows"]  # wahrscheinlich 5. Packen

parameter_range["shadows"]["default"] = 50
parameter_range["highlights"]["default"] = -50

parameter_range["exposure"]["min"] = -3  # wahrscheinlich 3. Packen
parameter_range["exposure"]["default"] = 0
parameter_range["exposure"]["max"] = 3

parameter_range["vibrance"]["min"] = 0
parameter_range["vibrance"]["default"] = 25
parameter_range["vibrance"]["max"] = 100

parameter_range["temperature"]["min"] = 1000
parameter_range["temperature"]["default"] = 6500
parameter_range["temperature"]["max"] = 12000

parameter_range["lcontrast"]["min"] = 0
parameter_range["lcontrast"]["default"] = 0  # i think default is impossible (possibly due to c bindings type conversions)
parameter_range["lcontrast"]["max"] = 40


def random_parameters() -> Tuple[str, Tuple[float, float]]:  # TODO redo for darktables
    change = random.choice(list(parameter_range.keys()))

    if change == "lcontrast":
        pos_neg = random.choice(["positive", "interval"])
        lcontrast_vals = [round(val, 1) for val in list(np.arange(0.1, 1, 0.1)) + list(range(1, 10)) + list(range(10, 41, 5))]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        if pos_neg == "positive":
            changeVal = (0, random.choice(lcontrast_vals))
        else:
            changeVal = (random.choice(lcontrast_vals), random.choice(lcontrast_vals))

    elif change == "hue":
        pos_neg = random.choice(["positive", "negative", "interval"])  # in order to not match a positive change with a negative one

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
        pos_neg = random.choice(["positive", "negative", "interval"])
        if pos_neg == "positive":
            changeVal = (parameter_range[change]["default"], random.choice(np.linspace(parameter_range[change]["default"], parameter_range[change]["max"], 10)))
        elif pos_neg == "negative":
            changeVal = (parameter_range[change]["default"], random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["default"], 10)))
        else:
            changeVal = (random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["max"], 20)), random.choice(np.linspace(parameter_range[change]["min"], parameter_range[change]["max"], 20)))
            # make sure to not compare an image to another one, which has been edited in the other "direction
            while (changeVal[0] < parameter_range[change]["default"] and changeVal[1] > parameter_range[change]["default"]) or (changeVal[0] > parameter_range[change]["default"] and changeVal[1] < parameter_range[change]["default"]):
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
