import argparse
import logging
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

from edit_image import edit_image, random_parameters

app = Flask(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


@app.route("/")
def survey():
    img = f"/img/{random.choice(app.imgs)}"
    edits = random_parameters()
    parameter, changes = edits[0], list(edits[1])
    shuffle(changes)
    leftChanges, rightChanges = changes
    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{parameter}:{changes}")  # TODO log hash and cookies
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
    logging.getLogger("forms").info(f"submit: {request.form.to_dict()}")  # TODO log cookies
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
