import json
import logging
import sys
from pathlib import Path
import time
import redis
from io import BytesIO

sys.path.insert(0, ".")
from edit_image import edit_image

r = redis.Redis(host="redis")


def preprocessImage():
    data = r.lpop("NIAA_img_q")
    if not data:
        return 0
    data = json.loads(data)
    logging.info(f"got: {data}")
    for i in range(10):
        img = r.hget("NIAA_img_q_prepared", data["img"])
        if img is not None:
            break
        logging.info("waiting for image")
        time.sleep(1)
    if img is None:
        return 1
    in_img = f"/tmp/in{Path(data['out']).suffix}"
    with open(in_img, "wb") as f:
        f.write(BytesIO(img).getbuffer())
    img = edit_image(img_path=in_img, change=data["parameter"], value=data["change"])  # keyword naming kann ich
    logging.info(f"finished image")

    if not img:
        raise ValueError(f"not working: {data}")

    img.save(data["out"])
    logging.info(f"saved image")
    return 1


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    while True:
        ret = preprocessImage()
        if ret == 0:
            logging.info(f"queue empty")
            break
