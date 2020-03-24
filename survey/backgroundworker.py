import json
import logging
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


def preprocessImage(name: int):
    logging.info(f"Thread {name}: starting")

    darktable_dir = f"/tmp/darktable/{name}"
    Path(darktable_dir).mkdir(parents=True, exist_ok=True)
    r = redis.Redis(host="survey-redis")

    while True:
        data = r.lpop("q")
        if not data:
            time.sleep(1)
            continue
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


def redis_to_sqlite():
    r = redis.StrictRedis(host="survey-redis")
    conn = sqlite3.connect("/data/logs/submissions.db", isolation_level=None)
    c = conn.cursor()

    added = 0
    while True:
        data = r.lpop("submissions")
        if not data:
            break
        data = json.loads(data)

        c.execute(  # databasenormali...what?
            """INSERT INTO submissions(loadTime,img,parameter,leftChanges,rightChanges,chosen,hashval,screenWidth,screenHeight,windowWidth,windowHeight,colorDepth,userid,usersubs,useragent) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data["loadTime"], data["img"], data["parameter"], data["leftChanges"], data["rightChanges"], data["chosen"], data["hashval"], data["screenWidth"], data["screenHeight"], data["windowWidth"], data["windowHeight"], data["colorDepth"], data["id"], data["count"], data["useragent"]),
        )

        added += 1
    conn.close()
    return added


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while True:  # um gestorbene Threads wieder zu starten
        if threading.activeCount() < 6:
            logging.info("Main    : creating one more thread")
            threading.Thread(target=preprocessImage, args=(threading.activeCount(),)).start()
            logging.info(f"Main    : {threading.activeCount()-1} Threads active")

        redis_to_sqlite()
        time.sleep(1)
