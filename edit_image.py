import collections
import math
import random
import subprocess
import tempfile
from pathlib import Path
from struct import pack, unpack
from typing import Dict, Tuple

import cv2
from jinja2 import Template
from PIL import Image


def edit_image(img_path: str, change: str, value: float, out_path: str = None, darktable_config: str = None) -> Image:
    if math.isclose(parameter_range[change]["default"], float(value)):
        print("default called")
        if out_path:
            Image.open(img_path).save(out_path)
            return out_path
        else:
            return Image.open(img_path)

    if "lcontrast" == change:  # XXX localcontrast xmp in darktable is broken atm. no idea why
        img = cv2.imread(img_path)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        cl = cv2.createCLAHE(clipLimit=float(value), tileGridSize=(8, 8)).apply(l)

        limg = cv2.merge((cl, a, b))
        if out_path:
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite(out_path, img)
            return out_path
        else:
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img)

    darktable_config_obj = None
    if not darktable_config:  # otherwise darktable can't open more than one instance in parallel
        darktable_config_obj = tempfile.TemporaryDirectory()
        darktable_config = darktable_config_obj.name

    edit_file = str(Path(darktable_config) / "edit.xmp")
    out_file = str(Path(darktable_config) / "out.jpg")

    if "contrast" == change or "brightness" == change or "saturation" == change:
        template_file = "./darktable_xmp/colisa.xmp"
        param_index = ["contrast", "brightness", "saturation"].index(change)
        default_str = "".join(["%02x" % b for b in bytearray(pack("f", 0))])
        change_val_enc = "".join(["%02x" % b for b in bytearray(pack("f", float(value)))])
        change_str = "".join([change_val_enc if _ == param_index else default_str for _ in range(3)])

    if "shadows" == change or "highlights" == change:
        template_file = "./darktable_xmp/shadhi.xmp"
        if "shadows" == change:
            change_str = f"000000000000c842{''.join(['%02x' % b for b in bytearray(pack('f', float(value)))])}000000000000c84200000000000048420000c842000048427f000000bd37863500000000"
        elif "highlights" == change:
            change_str = f"000000000000c8420000484200000000{''.join(['%02x' % b for b in bytearray(pack('f', float(value)))])}00000000000048420000c842000048427f000000bd37863500000000"

    if "exposure" == change:
        template_file = "./darktable_xmp/exposure.xmp"
        change_str = f"0000000000000000{''.join(['%02x' % b for b in bytearray(pack('f', float(value)))])}00004842000080c0"

    if "vibrance" == change:
        template_file = "./darktable_xmp/vibrance.xmp"
        change_str = "".join(["%02x" % b for b in bytearray(pack("f", float(value)))])

    if "temperature" == change or "tint" == change:
        template_file = "./darktable_xmp/temperature.xmp"
        change_str = parameter_range[change]["rangemapping"][value]

    with open(template_file) as template_file:
        Template(template_file.read()).stream(value=change_str).dump(edit_file)

    if out_path:
        subprocess.run(["darktable-cli", img_path, edit_file, out_path, "--core", "--library", ":memory:", "--configdir", darktable_config])
        if darktable_config_obj:
            darktable_config_obj.cleanup()
        return out_path
    else:
        subprocess.run(["darktable-cli", img_path, edit_file, out_file, "--core", "--library", ":memory:", "--configdir", darktable_config])
        img = Image.open(out_file)
        if darktable_config_obj:
            darktable_config_obj.cleanup()
        return img


parameter_range = collections.defaultdict(dict)
parameter_range["contrast"]["min"] = -1
parameter_range["contrast"]["default"] = 0
parameter_range["contrast"]["max"] = 1
parameter_range["contrast"]["range"] = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

parameter_range["brightness"] = parameter_range["contrast"]
parameter_range["saturation"] = parameter_range["contrast"]

parameter_range["shadows"]["min"] = -100  # wahrscheinlich 3. Packen
parameter_range["shadows"]["default"] = 50
parameter_range["shadows"]["max"] = 100
parameter_range["shadows"]["range"] = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


parameter_range["highlights"]["min"] = -100  # wahrscheinlich 5. Packen
parameter_range["highlights"]["default"] = -50
parameter_range["highlights"]["max"] = 100
parameter_range["highlights"]["range"] = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

parameter_range["exposure"]["min"] = -3  # wahrscheinlich 3. Packen
parameter_range["exposure"]["default"] = 0
parameter_range["exposure"]["max"] = 3
parameter_range["exposure"]["range"] = [-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]

parameter_range["vibrance"]["min"] = 0
parameter_range["vibrance"]["default"] = 25
parameter_range["vibrance"]["max"] = 100
parameter_range["vibrance"]["range"] = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

parameter_range["temperature"]["min"] = 1000
parameter_range["temperature"]["default"] = 6500
parameter_range["temperature"]["max"] = 25000
parameter_range["temperature"]["range"] = [2000, 3000, 4000, 5000, 6000, 6500, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]
parameter_range["temperature"]["rangemapping"] = {  # encoded RGB values, since darktable can't save temperature directly
    "2000": "6c71833e0000803f495cfd410000807f",
    "3000": "a623f43e0000803f9b9446400000807f",
    "4000": "e3832c3f0000803f705af13f0000807f",
    "5000": "f954543f0000803f9317ad3f0000807f",
    "6000": "ef39733f0000803f8c138b3f0000807f",
    "6500": "7405803f0000803fc301803f0000807f",
    "7000": "f39e853f0000803f72e86e3f0000807f",
    "8000": "17208f3f0000803ffed8553f0000807f",
    "9000": "c8c2963f0000803fdeb9443f0000807f",
    "10000": "d3029d3f0000803fd95f383f0000807f",
    "11000": "5035a23f0000803fcc172f3f0000807f",
    "12000": "9e96a63f0000803fb3e4273f0000807f",
    "13000": "ae52aa3f0000803f182a223f0000807f",
    "14000": "328aad3f0000803f68821d3f0000807f",
    "15000": "f555b03f0000803f08a9193f0000807f",
    "16000": "13c9b23f0000803f176e163f0000807f",
    "17000": "7ff2b43f0000803f37af133f0000807f",
    "18000": "10deb63f0000803f2753113f0000807f",
    "19000": "4295b83f0000803ff7460f3f0000807f",
    "20000": "c61fba3f0000803f2e7c0d3f0000807f",
    "21000": "e483bb3f0000803f93e70b3f0000807f",
    "22000": "cbc6bc3f0000803f50800a3f0000807f",
    "23000": "caecbd3f0000803f593f093f0000807f",
    "24000": "7ff9be3f0000803f021f083f0000807f",
    "25000": "f3efbf3f0000803fa91a073f0000807f",
}

parameter_range["tint"]["min"] = 0.2
parameter_range["tint"]["default"] = 1.0
parameter_range["tint"]["max"] = 2.3
parameter_range["tint"]["range"] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
parameter_range["tint"]["rangemapping"] = {  # encoded RGB values, since darktable can't save tint directly
    "0.2": "f16ad3bf0000803f79fd38420000807f",
    "0.3": "020b05c00000803f063824410000807f",
    "0.4": "0bee3ac00000803fb6e0af400000807f",
    "0.5": "cd4cabc00000803f804067400000807f",
    "0.6": "bce1b4c20000803fc2b926400000807f",
    "0.7": "c63ea9400000803f370cfd3f0000807f",
    "0.8": "d5ba18400000803fa425c63f0000807f",
    "0.9": "5d93ba3f0000803fc4469e3f0000807f",
    "1.0": "7405803f0000803fc301803f0000807f",
    "1.1": "da553a3f0000803f797c503f0000807f",
    "1.2": "13180c3f0000803fb52e2a3f0000807f",
    "1.3": "6c5ad63e0000803fa5a70a3f0000807f",
    "1.4": "5519a53e0000803fb980e03e0000807f",
    "1.5": "6cb97d3e0000803f31a2b33e0000807f",
    "1.6": "cea2403e0000803f8f098d3e0000807f",
    "1.7": "dfb60e3e0000803f1af8563e0000807f",
    "1.8": "df4eca3d0000803f41181c3e0000807f",
    "1.9": "8a08843d0000803f810cd03d0000807f",
    "2.0": "35ae0f3d0000803f9792663d0000807f",
    "2.1": "f6a21d3c0000803f1089803c0000807f",
    "2.2": "1f1f4fbc0000803f8250abbc000080ff",
    "2.3": "4c3404bd0000803f0c825dbd000080ff",
}

parameter_range["lcontrast"]["min"] = 0
parameter_range["lcontrast"]["default"] = 0
parameter_range["lcontrast"]["max"] = 40
parameter_range["lcontrast"]["range"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]


def random_parameters() -> Tuple[str, Tuple[float, float]]:
    change = random.choice(list(parameter_range.keys()))

    if change == "lcontrast":
        pos_neg = random.choice(["positive", "interval"])
    else:
        pos_neg = random.choice(["positive", "negative", "interval"])

    if pos_neg == "positive":
        changeVal = (parameter_range[change]["default"], random.choice([val for val in parameter_range[change]["range"] if val > parameter_range[change]["default"]]))
    elif pos_neg == "negative":
        changeVal = (parameter_range[change]["default"], random.choice([val for val in parameter_range[change]["range"] if val < parameter_range[change]["default"]]))
    else:
        changeVal = (random.choice(parameter_range[change]["range"]), random.choice(parameter_range[change]["range"]))
        # make sure to not compare an image to another one, which has been edited in the other "direction
        while (changeVal[0] < parameter_range[change]["default"] and changeVal[1] > parameter_range[change]["default"]) or (changeVal[0] > parameter_range[change]["default"] and changeVal[1] < parameter_range[change]["default"]):
            changeVal = (changeVal[0], random.choice(parameter_range[change]["range"]))

    return change, changeVal


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to a single image", default="/data/442284.jpg")
    parser.add_argument("--parameter", type=str, help="what to change: brightness, contrast...")
    parser.add_argument("--value", type=float, help="change value")
    parser.add_argument("--out", type=str, help="dest for edited images", default="/data/output.jpg")
    args = parser.parse_args()
    edit_image(img_path=args.image, change=args.parameter, value=args.value).save(args.out)

# darktable xmp crashkurs
# to encode
# "".join([ "%02x" % b for b in bytearray(pack("f", 10.0))])
# to decode
# unpack("f", b"\x00\x00\x96\x42")
# unpack("f", bytes.fromhex("00009642"))
