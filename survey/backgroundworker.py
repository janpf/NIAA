import argparse
import json
import logging
import secrets
import sqlite3
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import redis

sys.path.insert(0, ".")
from edit_image import edit_image

editedImageFolder = Path("/tmp/imgs/")


def preprocessImage():
    workerID = secrets.token_hex(nbytes=4)
    logging.info(f"worker {workerID} started")

    darktable_dir = f"/tmp/darktable/{workerID}"
    Path(darktable_dir).mkdir(parents=True, exist_ok=True)
    r = redis.Redis(host="redis")

    data = r.lpop("q")
    if not data:
        return
    data = json.loads(data)

    left = edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], darktable_config=darktable_dir)
    right = edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], darktable_config=darktable_dir)

    with BytesIO() as output:
        left.save(output, format="JPEG")
        leftImg = output.getvalue()

    with BytesIO() as output:
        right.save(output, format="JPEG")
        rightImg = output.getvalue()

    r.hmset("imgs", {f"{Path(data['img']).stem}_l.jpg": leftImg, f"{Path(data['img']).stem}_r.jpg": rightImg})
    r.rpush("pairs", json.dumps(data))
    r.close()


def redis_to_sqlite():
    r = redis.Redis(host="redis")
    conn = sqlite3.connect("/data/logs/submissions.db", isolation_level=None)
    c = conn.cursor()
    logging.info("connected to DBs")

    while True:
        data = r.lpop("submissions")
        if not data:
            break
        data = json.loads(data)
        logging.info(f"got: {data}")
        c.execute(  # databasenormali...what?
            """INSERT INTO submissions(loadTime,img,parameter,leftChanges,rightChanges,chosen,hashval,screenWidth,screenHeight,windowWidth,windowHeight,colorDepth,userid,usersubs,useragent) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data["loadTime"], data["img"], data["parameter"], data["leftChanges"], data["rightChanges"], data["chosen"], data["hashval"], data["screenWidth"], data["screenHeight"], data["windowWidth"], data["windowHeight"], data["colorDepth"], data["id"], data["count"], data["useragent"]),
        )

    conn.close()
    r.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, dest="job")
    args = parser.parse_args()

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while True:
        if args.job == "dbMover":
            redis_to_sqlite()
        elif args.job == "imagePreprocessor":
            preprocessImage()
        time.sleep(1)
