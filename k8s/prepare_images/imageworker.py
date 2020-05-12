import json
import logging
import sys
import time
from pathlib import Path

import redis

sys.path.insert(0, ".")
from edit_image import edit_image

r = redis.Redis(host="redis")


def preprocessImage():
    data = r.lpop("NIAA_img_q")
    if not data:
        return 0
    data = json.loads(data)

    logging.info(f"got: {data}")
    img = edit_image(img_path=data["img"], change=data["parameter"], value=data["change"])  # keyword naming kann ich
    logging.info(f"finished image")

    if not img:
        raise ValueError(f"not working: {data}")

    img.save(data["out"], format="JPEG")
    logging.info(f"saved image")
    return 1


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    while True:
        ret = preprocessImage()
        if ret == 0:
            logging.info(f"queue empty")
            break
        time.sleep(1)
