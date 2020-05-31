from pathlib import Path
from io import BytesIO
from PIL import Image
import sys
import logging
import redis

if len(sys.argv) > 1 and sys.argv[1] == "create":
    r = redis.Redis(host="localhost", port=7000)
    for parameter in Path("/scratch/stud/pfister/NIAA/pexels/edited_images").iterdir():
        for change in parameter.iterdir():
            indir = Path("/scratch/pexels/edited_images") / change.parts[-2] / change.parts[-1]
            r.rpush("corrupted_check", str(indir))
    exit()

r = redis.Redis(host="redis")

logging.getLogger().setLevel(logging.INFO)

dir_to_check = Path(r.lpop("corrupted_check").decode("ascii"))
log = open(f"/workspace/dataset_processing/corrupted/{dir_to_check.parts[-2]}_{dir_to_check.parts[-1]}.log", "a")
out = BytesIO()
imgs = []

for i, img in enumerate(dir_to_check.iterdir()):
    if i % 100 == 0:
        logging.info(f"{i} images done")
    try:
        im: Image = Image.open(img)
        im.save(out, format=img.suffix.replace(".", "").upper())
        im.close()
        del im
        out.seek(0)
    except:
        logging.info(img)
        print(img, file=log)
        log.flush()

logging.info("done")
