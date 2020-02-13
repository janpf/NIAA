import argparse
import logging
import math
import random
from io import BytesIO
from pathlib import Path
from random import shuffle
from typing import Dict, Tuple

import cv2
import numpy as np
import pillow_lut as lut
from flask import Flask, abort, redirect, render_template, request
from flask.helpers import send_file, url_for
from PIL import Image

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--imageFile", type=str, help="every line a file name", default="/scratch/stud/pfister/NIAA/pexels/train.txt")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images", default="/scratch/stud/pfister/NIAA/pexels/images")
parser.add_argument("--out", type=str, help="dir to log to", default="/scratch/stud/pfister/NIAA/pexels/logs")
args = parser.parse_args()

with open(args.imageFile, "r") as imgFile:
    imgs = [img.strip() for img in imgFile.readlines()]
    imgsSet = set(imgs)
poll_log = open(Path(args.out) / "poll.log", "a", buffering=1)


def edit_and_serve_image(img_path: str, changes: Dict[str, float]):
    if "lcontrast" in changes:
        img = cv2.imread(str(img_path))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(img_lab)

        cl = cv2.createCLAHE(clipLimit=changes["lcontrast"], tileGridSize=(8, 8)).apply(lab_planes[0])

        limg = cv2.merge(lab_planes)
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        _, buffer = cv2.imencode(".jpeg", img)
        img_io = BytesIO(buffer)

    else:
        img = Image.open(img_path)
        if "lcontrast" in changes:
            del changes["lcontrast"]
        img_io = BytesIO()
        img.convert("RGB").filter(lut.rgb_color_enhance(16, **changes)).save(img_io, "JPEG", quality=70)

    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


def random_parameters() -> Tuple[str, Tuple[float, float]]:
    parameters = {"brightness": [-1, 1], "exposure": [-5, 5], "contrast": [-1, 5], "warmth": [-1, 1], "saturation": [-1, 5], "vibrance": [-1, 5], "lcontrast": [0.1, 40]}  # TODO hue[0,1]
    if random.choice(["single"]) == "single":  # how many parameters to change at once # TODO add ability to change multiple parameters at once
        change = random.choice(list(parameters.keys()))
        if change == "lcontrast":
            pos_neg = random.choice(["positiv", "interval"])
            lcontrast_vals = [round(val, 1) for val in list(np.arange(0.1, 1, 0.1)) + list(range(1, 10)) + list(range(10, 41, 5))]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
            if pos_neg == "positive":
                changeVal = (0, random.choice(lcontrast_vals))
            else:
                changeVal = (random.choice(lcontrast_vals), random.choice(lcontrast_vals))
        else:
            pos_neg = random.choice(["positiv", "negative", "interval"])  # in order to not match a positive change with a negative one
            N = 1
            if pos_neg == "positive":
                changeVal = (0, round(random.uniform(0, parameters[change][1]), N))
            elif pos_neg == "negative":
                changeVal = (0, round(random.uniform(parameters[change][0], 0), N))
            else:
                changeVal = (round(random.uniform(parameters[change][0], parameters[change][1]), N), round(random.uniform(parameters[change][0], parameters[change][1]), N))
                while not math.copysign(1, changeVal[0]) == math.copysign(1, changeVal[1]):  # make sure to not compare an image to another one, which has been edited in the other "direction"
                    changeVal = (changeVal[0], round(random.uniform(parameters[change][0], parameters[change][1]), N))
    return change, changeVal


@app.route("/")
def survey():
    img = f"/img/{random.choice(imgs)}"
    edits = random_parameters()
    parameter, changes = edits[0], list(edits[1])
    shuffle(changes)
    leftChanges, rightChanges = changes
    print(f"{parameter}:{changes}")
    hashval = hash(f"{random.randint(0, 50000)}{img}{parameter}{leftChanges}{rightChanges}")
    return render_template("index.html", leftImage=f"{img}?{parameter}={leftChanges}&l&hash={hashval}", rightImage=f"{img}?{parameter}={rightChanges}&r&hash={hashval}", img=img, parameter=parameter, leftChanges=leftChanges, rightChanges=rightChanges, hash=hashval)


@app.route("/poll", methods=["POST"])
def poll():
    print(request.form.to_dict())
    print(request.form.to_dict(), file=poll_log)
    return redirect("/#left")


@app.route("/img/<image>")
def img(image: str):
    changes: Dict[str, float] = request.args.to_dict()
    changes = {k: float(v) for (k, v) in changes.items() if not k in ["r", "l", "hash"]}

    if not image in imgsSet:
        abort(404)
    return edit_and_serve_image(Path(args.imageFolder) / image, changes)


@app.before_request
def log_request_info():
    app.logger.debug("Body: %s", request.get_data())


if __name__ == "__main__":
    logging.basicConfig(filename=Path(args.out) / "debug.log", level=logging.DEBUG)

    app.run()
