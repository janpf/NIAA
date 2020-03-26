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

r = redis.Redis(host="redis")
conn = sqlite3.connect("/data/logs/submissions.db")
c = conn.cursor()


def preprocessImage():

    data = r.lpop("q")
    if not data:
        return
    data = json.loads(data)

    logging.info(f"got: {data}")

    left = edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"])
    right = edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"])

    if not left or not right:
        return

    with BytesIO() as output:
        left.save(output, format="JPEG")
        leftImg = output.getvalue()

    with BytesIO() as output:
        right.save(output, format="JPEG")
        rightImg = output.getvalue()

    r.hmset("imgs", {f"{Path(data['img']).stem}_l.jpg": leftImg, f"{Path(data['img']).stem}_r.jpg": rightImg})
    r.rpush("pairs", json.dumps(data))


def redis_to_sqlite():
    while True:
        data = r.lpop("submissions")
        if not data:
            break
        data = json.loads(data)
        logging.info(f"got: {data}")
        c.execute(  # databasenormali...what?
            """INSERT INTO submissions(loadTime,submitTime,img,parameter,leftChanges,rightChanges,chosen,hashval,screenWidth,screenHeight,windowWidth,windowHeight,colorDepth,userid,usersubs,useragent) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data["loadTime"],
                data["submitTime"],
                data["img"],
                data["parameter"],
                data["leftChanges"],
                data["rightChanges"],
                data["chosen"],
                data["hashval"],
                data["screenWidth"],
                data["screenHeight"],
                data["windowWidth"],
                data["windowHeight"],
                data["colorDepth"],
                data["id"],
                data["count"],
                data["useragent"],
            ),
        )
        conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str)
    args = parser.parse_args()

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while True:
        if args.job == "dbMover":
            redis_to_sqlite()
        elif args.job == "imagePreprocessor":
            preprocessImage()
        time.sleep(1)
