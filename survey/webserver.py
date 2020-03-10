import logging
import random
import secrets
import sqlite3
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, session
from flask.helpers import send_file, url_for

from edit_image import edit_image, random_parameters

app = Flask(__name__)
# TODO darktable in feste dirs


@app.route("/")
def survey():
    conn = sqlite3.connect(app.config["queueDB"], isolation_level="EXCLUSIVE")
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN EXCLUSIVE")  # completely locks down database for all other accesses
    c = conn.cursor()
    data = c.execute("""SELECT * FROM queue ORDER BY id LIMIT 1""").fetchone()  # first inserted imagepair
    c.execute("""DELETE FROM queue WHERE id = ?""", (data["id"],))
    conn.commit()
    conn.close()
    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{data['parameter']}:{[data['leftChanges'], data['rightChanges']]}; {session}")
    return render_template("index.html", count=session["count"], **data)


@app.route("/poll", methods=["POST"])
def poll():
    data = request.form.to_dict()
    logging.getLogger("forms").info(f"submit: {data}; {session}")

    conn = sqlite3.connect(app.config["subDB"])
    c = conn.cursor()

    c.execute(  # databasenormali...what?
        """INSERT INTO submissions(img,parameter,leftChanges,rightChanges,chosen,hashval,screenWidth,screenHeight,windowWidth,windowHeight,colorDepth,userid,usersubs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (data["img"], data["parameter"], data["leftChanges"], data["rightChanges"], data["chosen"], data["hashval"], data["screenWidth"], data["screenHeight"], data["windowWidth"], data["windowHeight"], data["colorDepth"], session["id"], session["count"]),
    )

    conn.commit()
    conn.close()

    session["count"] += 1
    return redirect("/#left")


@app.route("/img/<image>")
def img(image: str):
    changes: Dict[str, float] = request.args.to_dict()

    if not image in app.imgsSet:
        abort(404)

    edited = image.split(".")[0] + f"_{changes['side']}.jpg"  # only works if one dot in imagepath :D

    max_wait = 20
    while not Path(edited).exists() and max_wait > 0:  # sometimes darktable takes reeeeaaally long
        max_wait -= 1
        time.sleep(1)

    return send_file(app.config.get("editedImageFolder") / edited, mimetype="image/jpeg")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authorized", False):
        return redirect(url_for("survey"))

    if not session.get("id", None):
        session["id"] = secrets.token_hex(nbytes=16)

    if not session.get("count", None):
        session["count"] = 0

    if request.method == "POST":  # "Ich habe verstanden"
        session["authorized"] = True
        return redirect(url_for("survey"))

    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/preprocess")
def preprocessImages():
    conn = sqlite3.connect(app.config["queueDB"], isolation_level=None)
    c = conn.cursor()
    queueRanEmpty = False
    try:
        c.execute("""SELECT COUNT(*) FROM queue""").fetchone()[0]
    except:
        c.execute("""INSERT INTO queue(img,parameter,leftChanges,rightChanges,hashval) VALUES (?,?,?,?,?)""", ("tmp", "tmp", "tmp", "tmp", "tmp"))
        queueRanEmpty = True

    while c.execute("""SELECT COUNT(*) FROM queue""").fetchone()[0] < 50:  # preprocess up to 50 imagepairs, if less than 30 are already preprocessed
        chosen_img = random.choice(app.imgs)
        image_file = Path(app.config.get("imageFolder")) / chosen_img
        img = f"/img/{chosen_img}"

        edits = random_parameters()
        parameter, changes = edits[0], list(edits[1])
        random.shuffle(changes)
        leftChanges, rightChanges = changes
        hashval = str(hash(f"{random.randint(0, 50000)}{img}{parameter}{leftChanges}{rightChanges}"))

        data = (img, parameter, leftChanges, rightChanges, hashval)
        c.execute("""INSERT INTO queue(img,parameter,leftChanges,rightChanges,hashval) VALUES (?,?,?,?,?)""", data)

        Process(target=edit_image, args=(str(image_file), parameter, leftChanges, str(app.config.get("editedImageFolder") / f"{image_file.stem}_l.jpg"))).start()
        Process(target=edit_image, args=(str(image_file), parameter, rightChanges, str(app.config.get("editedImageFolder") / f"{image_file.stem}_r.jpg"))).start()

    if queueRanEmpty:
        c.execute("""DELETE FROM queue WHERE hashval = tmp""")
    conn.close()

    subprocess.Popen(f"ls -tp {app.config.get('editedImageFolder')} | grep -v '/$' | tail -n +201 | xargs -d '\n' -r rm --", shell=True)  # only keep 200 latest images
    return ""


@app.before_request
def log_request_info():
    rlogger = logging.getLogger("requests")
    rlogger.info("Headers: %s", request.headers)
    rlogger.info("Session: %s", session)
    if not session.get("authorized", False) and not (request.endpoint == "login" or request.endpoint == "preprocess"):
        return redirect(url_for("login"))


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def load_app(imgFile: str = "/data/train.txt", imageFolder: str = "/data/images", out: str = "/data/logs") -> Flask:  # for gunicorn: https://github.com/benoitc/gunicorn/issues/135
    out = Path(out)
    # all variables will be forked, but not synchronized between gunicorn threads
    logging.basicConfig(filename=out / "flask.log", level=logging.DEBUG)
    app.logger.handlers.extend(logging.getLogger("gunicorn.error").handlers)
    app.logger.handlers.extend(logging.getLogger("gunicorn.warning").handlers)
    app.logger.setLevel(logging.DEBUG)

    setup_logger("compares", out / "compares.log")
    setup_logger("forms", out / "submissions.log")
    setup_logger("requests", out / "requests.log")

    app.config["SERVER_NAME"] = None

    app.config["imageFolder"] = imageFolder
    app.config["editedImageFolder"] = Path("/tmp/imgs/")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["queueDB"] = out / "queue.db"
    app.config["subDB"] = out / "submissions.db"

    app.config.get("editedImageFolder").mkdir(parents=True, exist_ok=True)
    app.secret_key = "secr3t"  # XXX

    with open(imgFile, "r") as f:
        app.imgs = [img.strip() for img in f.readlines()]
        app.imgsSet = set(app.imgs)
    return app
