import argparse
import random
from io import BytesIO
from pathlib import Path
import math

import pillow_lut as lut
from flask import Flask, abort, render_template, request
from flask.helpers import send_file
from PIL import Image
from skimage import exposure, io

app = Flask(__name__)  # TODO logging


parser = argparse.ArgumentParser()
parser.add_argument("--imageFile", type=str, help="every line a file name", default="/scratch/stud/pfister/NIAA/pexels/train.txt")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images", default="/scratch/stud/pfister/NIAA/pexels/images")
parser.add_argument("--out", type=str, help="dir to log to", default="/scratch/stud/pfister/NIAA/pexels/survey.log")
args = parser.parse_args()

with open(args.imageFile, "r") as imgFile:
    imgs = [img.strip() for img in imgFile.readlines()]
    imgsSet = set(imgs)


def edit_and_serve_image(img_path, changes):  # TODO be able to apply lcontrast in addition to other parameter changes
    img_io = BytesIO()

    if changes["lcontrast"] != 0:
        img = io.imread(img_path)
        clip_limit = 0.03
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        io.imsave(img_io, img_adapteq)  # Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.

    else:
        img = Image.open(img_path)
        changes.pop("lcontrast")
        img_filter = lut.rgb_color_enhance(16, **changes)
        img.filter(img_filter).save(img_io, "JPEG", quality=70)

    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


def random_parameters():
    modes = ["single"]  # how many parametes to change at once
    parameters = {"brightness": [-1, 1], "exposure": [-5, 5], "contrast": [-1, 5], "warmth": [-1, 1], "saturation": [-1, 5], "vibrance": [-1, 5]}  # TODO hue and lcontrast
    if random.choice(modes) == "single":
        pos_neg = random.choice(["positiv", "negative", "interval"])  # in order to not match a positive change with a negative one
        change = random.choice(list(parameters.keys()))
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
    img = random.choice(imgs)
    img = f"/img/{img}"
    edits = random_parameters()
    print(edits)
    return render_template("index.html", leftImage=f"{img}?{edits[0]}={edits[1][0]}", rightImage=f"{img}?{edits[0]}={edits[1][1]}")


@app.route("/poll")
def poll():
    return "not yet implemented"


@app.route("/img/<image>")
def img(image):
    print(f"{Path(args.imageFolder) / image} requested")

    # brightness – One value for all channels, or tuple of three values from -1.0 to 1.0. Use exposure for better result.
    # exposure – One value for all channels, or tuple of three values from -5.0 to 5.0.
    # contrast – One value for all channels, or tuple of three values from -1.0 to 5.0.
    # warmth – One value from -1.0 to 1.0.
    # saturation – One value for all channels, or tuple of three values from -1.0 to 5.0.
    # vibrance – One value for all channels, or tuple of three values from -1.0 to 5.0.
    # hue – One value from 0 to 1.0.
    # lcontrast
    changes = {}
    changes["brightness"] = request.args.get("brightness", default=0, type=float)
    changes["exposure"] = request.args.get("exposure", default=0, type=float)
    changes["contrast"] = request.args.get("contrast", default=0, type=float)
    changes["warmth"] = request.args.get("warmth", default=0, type=float)
    changes["saturation"] = request.args.get("saturation", default=0, type=float)
    changes["vibrance"] = request.args.get("vibrance", default=0, type=float)
    changes["hue"] = request.args.get("hue", default=0, type=float)
    changes["lcontrast"] = request.args.get("lcontrast", default=0, type=float)

    if not image in imgsSet:
        abort(404)
    return edit_and_serve_image(Path(args.imageFolder) / image, changes)


if __name__ == "__main__":
    app.run()
