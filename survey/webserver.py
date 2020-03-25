import json
import logging
import random
import secrets
import time
from io import BytesIO
from pathlib import Path

import redis
from flask import Flask, abort, g, redirect, render_template, request, session
from flask.helpers import send_file, url_for

from edit_image import random_parameters

app = Flask(__name__)


@app.route("/")
def survey():
    data = g.r.lpop("pairs")
    data = json.loads(data)

    logging.getLogger("compares").info(f"{session.get('name', 'Unknown')}:{data['img']}:{data['parameter']}:{[data['leftChanges'], data['rightChanges']]}; {session}")
    return render_template("index.html", count=session["count"], img=f"/img/{Path(data['img']).name}", parameter=data["parameter"], leftChanges=data["leftChanges"], rightChanges=data["rightChanges"], hashval=data["hashval"], loadTime=time.strftime("%Y-%m-%d %H:%M:%S"))


@app.route("/poll", methods=["POST"])
def poll():
    data = request.form.to_dict()
    logging.getLogger("forms").info(f"submit: {data}; {session}")
    data = {
        "loadTime": data["loadTime"],
        "submitTime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "img": data["img"],
        "parameter": data["parameter"],
        "leftChanges": data["leftChanges"],
        "rightChanges": data["rightChanges"],
        "chosen": data["chosen"],
        "hashval": data["hashval"],
        "screenWidth": data["screenWidth"],
        "screenHeight": data["screenHeight"],
        "windowWidth": data["windowWidth"],
        "windowHeight": data["windowHeight"],
        "colorDepth": data["colorDepth"],
        "id": session["id"],
        "count": session["count"],
        "useragent": request.headers.get("User-Agent"),
    }
    session["count"] += 1
    g.r.rpush("submissions", json.dumps(data))
    return redirect("/#left")


@app.route("/img/<image>")
def img(image: str):
    if not image in app.imgsSet:
        abort(404)

    changes: Dict[str, float] = request.args.to_dict()
    edited = image.split(".")[0] + f"_{changes['side']}.jpg"  # only works if one dot in imagepath :D
    img = g.r.hmget("imgs", edited)[0]  # should only be one
    g.r.hdel("imgs", edited)

    img = BytesIO(img)
    img.seek(0)
    return send_file(img, mimetype="image/jpeg")


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
    while g.r.llen("q") + g.r.llen("pairs") <= 500:  # preprocess up to 1000 imagepairs

        newPairs = []
        for _ in range(50):
            chosen_img = random.choice(app.imgs)
            image_file = Path(app.config.get("imageFolder")) / chosen_img

            edits = random_parameters()
            parameter, changes = edits[0], list(edits[1])
            random.shuffle(changes)
            leftChanges, rightChanges = changes
            hashval = str(hash(f"{random.randint(0, 50000)}{chosen_img}{parameter}{leftChanges}{rightChanges}"))
            newPairs.append({"img": str(Path(app.config["imageFolder"]) / chosen_img), "parameter": parameter, "leftChanges": leftChanges, "rightChanges": rightChanges, "hashval": hashval})

        newPairs = [json.dumps(val) for val in newPairs]
        g.r.rpush("q", *newPairs)
    return "ok"


@app.before_request
def before_request():
    if "kube-probe" in request.headers.get("User-Agent"):
        return "all good. kthxbai"
    rlogger = logging.getLogger("requests")
    rlogger.info("Headers: %s", request.headers)
    rlogger.info("Session: %s", session)
    g.r = redis.Redis(host="redis")  # type: redis.Redis
    # if (not session.get("authorized", False)) and not (request.endpoint == "login" or request.endpoint == "preprocess"): # FIXME
    #    return redirect(url_for("login"))


@app.after_request
def after_request(response):
    try:
        if g.r is not None:
            g.r.close()
    except:
        pass  # dann halt nicht

    return response


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

    app.config.get("editedImageFolder").mkdir(parents=True, exist_ok=True)
    app.secret_key = "secr3t"  # WONTFIX

    with open(imgFile, "r") as f:
        app.imgs = [img.strip() for img in f.readlines()]
        app.imgsSet = set(app.imgs)
    return app
