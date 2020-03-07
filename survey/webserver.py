import argparse
import logging
import random
import secrets
import tempfile
from collections import deque
from io import BytesIO
from multiprocessing import Lock, Process, SimpleQueue
from pathlib import Path
from random import shuffle
from typing import Any, Dict, Tuple

from flask import Flask, abort, redirect, render_template, request, session
from flask.helpers import send_file, url_for

from edit_image import edit_image, random_parameters
import sqlite3

app = Flask(__name__)

# TODO just write and log everything into a sqlite


def preprocessImages():  # TODO cleanup # TODO move to iframe or sth
    conn = sqlite3.connect(app.config["queueDB"])
    c = conn.cursor()
    queueRanEmpty = False
    try:
        c.execute("""SELECT * FROM queue""").fetchone()[0]
    except:
        c.execute("""INSERT INTO queue VALUES (?,?,?,?,?)""", ("tmp", "tmp", "tmp", "tmp", "tmp"))
        queueRanEmpty = True

    if c.execute("""SELECT * FROM queue""").fetchone()[0] < 30:
        while c.execute("""SELECT * FROM queue""").fetchone()[0] < 40:  # preprocess 40 imagepairs, if less than 30 are already preprocessed
            chosen_img = random.choice(app.imgs)
            image_file = Path(app.config.get("imageFolder")) / chosen_img
            img = f"/img/{chosen_img}"

            edits = random_parameters()
            parameter, changes = edits[0], list(edits[1])
            shuffle(changes)
            leftChanges, rightChanges = changes
            hashval = str(hash(f"{random.randint(0, 50000)}{img}{parameter}{leftChanges}{rightChanges}"))

            data = (img, parameter, leftChanges, rightChanges, hashval)
            c.execute("""INSERT INTO queue VALUES (?,?,?,?,?)""", data)
            conn.commit()  # instacommit, otherwise other threads could drastically overfill the queue

            Process(target=edit_image, args=(str(image_file), parameter, leftChanges, str(app.config.get("editedImageFolder") / f"{image_file.stem}_l.jpg"))).start()
            Process(target=edit_image, args=(str(image_file), parameter, rightChanges, str(app.config.get("editedImageFolder") / f"{image_file.stem}_r.jpg"))).start()

    if queueRanEmpty:
        c.execute("""DELETE FROM queue WHERE img = tmp""")
    conn.commit()
    conn.close()


@app.route("/")
def survey():
    preprocessImages()  # queue new images for preprocessing

    conn = sqlite3.connect(app.config["queueDB"])
    c = conn.cursor()
    data = c.execute("""SELECT min(id) FROM queue""").fetchone()  # first inserted imagepair
    conn.close()

    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{data['parameter']}:{[data['leftChanges'], data['rightChanges']]}; {session}")
    return render_template("index.html", username=session["name"], count=session["count"], **data)


@app.route("/poll", methods=["POST"])
def poll():
    print(request.form.to_dict())
    logging.getLogger("forms").info(f"submit: {request.form.to_dict()}; {session}")
    session["count"] += 1
    return redirect("/#left")


@app.route("/img/<image>")  # TODO wait for first image
def img(image: str):
    changes: Dict[str, float] = request.args.to_dict()

    if not image in app.imgsSet:
        abort(404)
    edited = image.split(".")[0] + f"_{changes['side']}.jpg"  # only works if one dot in imagepath :D
    return send_file(app.config.get("editedImageFolder") / edited, mimetype="image/jpeg")


@app.route("/login", methods=["GET", "POST"])
def login():
    preprocessImages()  # queue new images for preprocessing
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
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
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
    app.config["editedImageFolder"] = Path("/tmp/imgs/")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["queueDB"] = "/data/logs/queue.db"

    app.config.get("editedImageFolder").mkdir(parents=True, exist_ok=True)
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
