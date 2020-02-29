import argparse
import logging
import random
import secrets
import tempfile
from io import BytesIO
from multiprocessing import Lock, Process, SimpleQueue
from pathlib import Path
from random import shuffle
from typing import Dict, Tuple

from flask import Flask, abort, redirect, render_template, request, session
from flask.helpers import send_file, url_for

from edit_image import edit_image_mp, random_parameters

app = Flask(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

dictLock = Lock()
queuedImageData = dict()  # hashval to dict
preprocessedImages = dict()  # hashval to tuple of queues which hold exactly one image each


def preprocessImages():
    with dictLock:
        while len(queuedImageData) < 10:
            chosen_img = random.choice(app.imgs)
            image_file = Path(app.config.get("imageFolder")) / chosen_img
            img = f"/img/{chosen_img}"

            edits = random_parameters()
            parameter, changes = edits[0], list(edits[1])
            shuffle(changes)
            leftChanges, rightChanges = changes
            hashval = hash(f"{random.randint(0, 50000)}{img}{parameter}{leftChanges}{rightChanges}")

            queuedImageData[hashval] = {"img": img, "edits": edits, "parameter": parameter, "leftChanges": leftChanges, "rightChanges": rightChanges, "hashval": hashval}
            preprocessedImages[hashval] = (SimpleQueue(), SimpleQueue())

            Process(target=edit_image_mp, args=(str(image_file), parameter, leftChanges, preprocessedImages[hashval][0])).start()
            Process(target=edit_image_mp, args=(str(image_file), parameter, rightChanges, preprocessedImages[hashval][1])).start()


@app.route("/")
def survey():
    preprocessImages()  # queue new images for preprocessing
    with dictLock:
        first_hash = list(queuedImageData)[0]  # get first queued image (hopefully)
        data = queuedImageData[first_hash]
        del queuedImageData[first_hash]  # so that no other "/index" call can get the same comparison

    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{data['parameter']}:{data['changes']}")  # TODO log hash and cookies
    return render_template("index.html", username=session["name"], count=session["count"], **data)


@app.route("/poll", methods=["POST"])
def poll():
    print(request.form.to_dict())
    logging.getLogger("forms").info(f"submit: {request.form.to_dict()}")  # TODO log cookies
    session["count"] += 1
    return redirect("/#left")


@app.route("/img/<image>")  # XXX naming is a minefield in here
def img(image: str):
    changes: Dict[str, float] = request.args.to_dict()

    if not image in app.imgsSet:
        abort(404)

    with dictLock:
        if "l" in changes:
            img = preprocessedImages[changes["hash"]][0].get()
            preprocessedImages[changes["hash"]] = (None, preprocessedImages[changes["hash"]][1])
        else:
            img = preprocessedImages[changes["hash"]][1].get()
            preprocessedImages[changes["hash"]] = (preprocessedImages[changes["hash"]][0], None)

        if preprocessedImages[changes["hash"]][0] is None and preprocessedImages[changes["hash"]][1] is None:  # delete images from the queue, if both have already been served
            del preprocessedImages[changes["hash"]]

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
