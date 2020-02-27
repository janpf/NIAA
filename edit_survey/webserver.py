import argparse
import logging
import math
import random
import secrets
import tempfile
from io import BytesIO
from pathlib import Path
from random import shuffle
from typing import Dict, Tuple

import numpy as np
from flask import Flask, abort, redirect, render_template, request, session
from flask.helpers import send_file, url_for

from edit_image import edit_image, parameter_range

app = Flask(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def random_parameters() -> Tuple[str, Tuple[float, float]]: # TODO verteilungen gefallen mir nicht so # TODO bessere hunderter runden

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
            changeVal = (0, random.choice(np.arange(1, 11, 1)))  # TODO check
        elif pos_neg == "negative":
            changeVal = (0, random.choice(np.arange(-10, 0, 1)))  # TODO check
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


@app.route("/")
def survey():
    img = f"/img/{random.choice(app.imgs)}"
    edits = random_parameters()
    parameter, changes = edits[0], list(edits[1])
    shuffle(changes)
    leftChanges, rightChanges = changes
    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{parameter}:{changes}") # TODO log hash and cookies
    # print(f"{parameter}:{changes}")
    hashval = hash(f"{random.randint(0, 50000)}{img}{parameter}{leftChanges}{rightChanges}")
    # fmt: off
    return render_template(
        "index.html",
        leftImage=f"{img}?{parameter}={leftChanges}&l&hash={hashval}",
        rightImage=f"{img}?{parameter}={rightChanges}&r&hash={hashval}",
        img=img, parameter=parameter,
        leftChanges=leftChanges,
        rightChanges=rightChanges,
        hash=hashval,
        username=session["name"],
        count=session["count"]
    )
    # fmt: on


@app.route("/poll", methods=["POST"])
def poll():
    print(request.form.to_dict())
    logging.getLogger("forms").info(f"submit: {request.form.to_dict()}") # TODO log cookies
    session["count"] += 1
    return redirect("/#left")


@app.route("/img/<image>")  # XXX naming is a minefield in here
def img(image: str):
    changes: Dict[str, float] = request.args.to_dict()
    changes = {k: float(v) for (k, v) in changes.items() if not k in ["r", "l", "hash"]}

    if not image in app.imgsSet:
        abort(404)
    if len(changes) != 1:
        abort(500)

    image_file = Path(app.config.get("imageFolder")) / image
    change, value = changes.popitem()

    img = edit_image(str(image_file), change, value)
    file_object = BytesIO()
    img.save(file_object, "JPEG")
    file_object.seek(0)

    return send_file(file_object, mimetype="image/jpeg")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authorized", False):
        return redirect(url_for("survey"))

    if not session.get("id", None):
        session["id"] = secrets.token_hex(nbytes=16)

    if not session.get("name", None):
        session["name"] = f"Anon#{secrets.token_hex(nbytes=4)}"

    if not session.get("count", None):
        session["count"] = 0

    if request.method == "POST":
        data = request.form.to_dict()
        if data["username"]:
            session["name"] = data["username"]

        if data["password"] == "lala":
            session["authorized"] = True
            return redirect(url_for("survey"))

    return render_template("login.html", username=session["name"])


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.before_request
def log_request_info():
    rlogger = logging.getLogger("requests")
    rlogger.info("Body: %s", request.get_data())
    rlogger.info("Headers: %s", request.headers)
    rlogger.info("Session: %s", session)
    if not session.get("authorized", False) and request.endpoint != "login":
        return redirect(url_for("login"))


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def load_app(imgFile="/data/train.txt", imageFolder="/data/images", out="/data/logs"):  # for gunicorn # https://github.com/benoitc/gunicorn/issues/135

    logging.basicConfig(filename=Path(out) / "flask.log", level=logging.DEBUG)
    app.logger.handlers.extend(logging.getLogger("gunicorn.error").handlers)
    app.logger.handlers.extend(logging.getLogger("gunicorn.warning").handlers)
    app.logger.setLevel(logging.DEBUG)

    setup_logger("compares", Path(out) / "compares.log")
    setup_logger("forms", Path(out) / "forms.log")
    setup_logger("requests", Path(out) / "requests.log")

    app.config["imageFolder"] = imageFolder
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    app.secret_key = "secr3t"  # TODO

    with open(imgFile, "r") as f:
        app.imgs = [img.strip() for img in f.readlines()]
        app.imgsSet = set(app.imgs)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageFile", type=str, help="every line a file name", default="/scratch/stud/pfister/NIAA/pexels/train.txt")
    parser.add_argument("--imageFolder", type=str, help="path to a folder of images", default="/scratch/stud/pfister/NIAA/pexels/images")
    parser.add_argument("--out", type=str, help="path to log to", default="/scratch/stud/pfister/NIAA/pexels/logs")
    args = parser.parse_args()

    load_app(args.imageFile, args.imageFolder, args.out)

    app.run(debug=True)
