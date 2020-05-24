import json
import logging
from pathlib import Path
from io import BytesIO
import redis
from PIL import Image

r = redis.Redis(host="redis")
pipe = r.pipeline()


def preprocessImage():
    data = r.lrange("NIAA_img_q", 1, 5000)
    data = [json.loads(val)["img"] for val in data]
    done = list(r.hkeys("NIAA_img_q_prepared"))

    for val in data:
        if val in done:
            continue
        logging.info(val)
        img = Image.open(val)
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img = output.getvalue()
        pipe.hset("NIAA_img_q_prepared", key=val, value=img)
        done.append(val)


def clearOldImages():
    done = r.hkeys("NIAA_img_q_prepared")
    data = r.lrange("NIAA_img_q", 1, 5000)
    data = [json.loads(val)["img"] for val in data]

    for key in done:
        if not key in data:
            pipe.hdel("NIAA_img_q_prepared", key)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    while True:
        logging.info(f"{r.hlen('NIAA_img_q_prepared')} images queued")
        preprocessImage()
        pipe.execute()
        logging.info(f"{r.hlen('NIAA_img_q_prepared')} images queued")
        clearOldImages()
        pipe.execute()
