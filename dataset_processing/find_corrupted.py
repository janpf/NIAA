from pathlib import Path
from io import BytesIO
from PIL import Image
import sys
import logging
import redis

logging.getLogger().setLevel(logging.INFO)

out = BytesIO()


def check(img: Path):
    im: Image = Image.open(img)
    img_format = Path(img).suffix.replace(".", "").upper()  # "epic_Image.pNg" => "PNG"
    if img_format == "JPG":
        img_format = "JPEG"
    im.save(out, format=img_format)
    im.close()
    del im
    out.seek(0)


if len(sys.argv) > 1 and sys.argv[1] == "create":
    r = redis.Redis(host="localhost", port=7000)
    for parameter in Path("/scratch/stud/pfister/NIAA/pexels/edited_images").iterdir():
        for change in parameter.iterdir():
            indir = Path("/scratch/pexels/edited_images") / change.parts[-2] / change.parts[-1]
            r.rpush("corrupted_check", str(indir))
    exit()

if len(sys.argv) > 1 and sys.argv[1] == "recheck_create":
    r = redis.Redis(host="localhost", port=7000)
    pipe = r.pipeline()
    for f in Path("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/corrupted").iterdir():
        with open(f, "r") as f:
            for img in f.readlines():
                pipe.rpush("corrupted_recheck", img.strip())
    pipe.execute()
    exit()

r = redis.Redis(host="redis")

if len(sys.argv) > 1 and sys.argv[1] == "recheck":
    while True:
        try:
            img = Path(r.lpop("corrupted_recheck").decode("ascii"))
        except:
            logging.info("queue empty")
            exit()

        try:
            check(img)
        except:
            logging.info(img)
            r.rpush("still_corrupted", img)

dir_to_check = Path(r.lpop("corrupted_check").decode("ascii"))
log = open(f"/workspace/dataset_processing/corrupted/{dir_to_check.parts[-2]}_{dir_to_check.parts[-1]}.log", "a")
imgs = []

for i, img in enumerate(dir_to_check.iterdir()):
    if i % 100 == 0:
        logging.info(f"{i} images done")
    try:
        check(img)
    except:
        logging.info(img)
        print(img, file=log)
        log.flush()

logging.info("done")
