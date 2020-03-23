import json
import logging
import sys
import threading
import time
from pathlib import Path

import redis

sys.path.insert(0, ".")
from edit_image import edit_image


queueDB = "/data/logs/queue.db"
editedImageFolder = Path("/tmp/imgs/")


def preprocessImage(name: int, q: SimpleQueue):
    logging.info(f"Thread {name}: starting")

    darktable_dir = f"/tmp/darktable/{name}"
    Path(darktable_dir).mkdir(parents=True, exist_ok=True)
    r = redis.StrictRedis(host="survey-redis")

    while True:
        data = r.lpop("q")
        if not data:
            time.sleep(1)
            continue
        data = json.loads(data)

        left = edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], darktable_config=darktable_dir)
        right = edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], darktable_config=darktable_dir)

        with io.BytesIO() as output:
            left.save(output, format="GIF")
            leftImg = output.getvalue()

        with io.BytesIO() as output:
            right.save(output, format="GIF")
            rightImg = output.getvalue()
        r.hmset("imgs", {f"{Path(data['img']).stem}_l.jpg": leftImg, f"{Path(data['img']).stem}_r.jpg": rightImg})  # FIXME encode images


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while True:  # um gestorbene Threads wieder zu starten
        if threading.activeCount() < 6:
            logging.info("Main    : creating one more thread")
            threading.Thread(target=preprocessImage, args=(threading.activeCount(),)).start()
            logging.info(f"Main    : {threading.activeCount()-1} Threads active")

        time.sleep(1)  # FIXME commit redis to sqlite
