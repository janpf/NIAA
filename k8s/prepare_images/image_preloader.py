import json
import logging
from pathlib import Path
import redis

r = redis.Redis(host="redis")


def preprocessImage():
    data = r.lrange("NIAA_img_q")
    if not data:
        return 0
    data = json.loads(data)

    logging.info(f"got: {data}")
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
