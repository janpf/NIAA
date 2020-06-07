import json
import math
import sys
from pathlib import Path

from PIL import Image
import redis

sys.path.insert(0, ".")
from edit_image import create_xmp_file, parameter_range

r = redis.Redis(host="localhost", port=7000)
pipe = r.pipeline()

pexels_dir = Path("/scratch") / "stud" / "pfister" / "NIAA" / "pexels"
img_dir = pexels_dir / "images"
out_dir = pexels_dir / "edited_images"


pexels_docker_dir = Path("/scratch") / "pexels"
img_docker_dir = pexels_docker_dir / "images"
out_docker_dir = pexels_docker_dir / "edited_images"

mode = ["all", "missing", "repair", "png"][1]

del parameter_range["lcontrast"]

if mode == "missing":
    orig_imgs = list(img_dir.iterdir())
    orig_imgs: set = {str(img.name) for img in orig_imgs}

    for parameter in parameter_range:
        print(parameter)
        for change in parameter_range[parameter]["range"]:
            if math.isclose(change, parameter_range[parameter]["default"]):
                continue
            edited_imgs = list((out_dir / parameter / str(change)).iterdir())
            edited_imgs: set = {str(img.name) for img in edited_imgs}
            missing = orig_imgs.difference(edited_imgs)
            print(parameter, change, "missing:", len(missing))
            for image in missing:
                image = Path(image)
                data = {"img": str(img_docker_dir / image.name), "parameter": parameter, "change": change, "out": str(out_docker_dir / parameter / str(change) / image.name)}
                pipe.rpush("NIAA_img_q", json.dumps(data))
        pipe.execute()

elif mode == "repair":
    broken = []
    for f in Path("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/corrupted").iterdir():
        with open(f, "r") as f:
            broken.extend(f.readlines())
    for img_path in broken:
        img_path = Path(img_path)
        img_name = img_path.name.strip()
        parameter, change = img_path.parts[-3:-1]
        for tmp in parameter_range[parameter]["range"]:
            if math.isclose(float(change), tmp):
                change = tmp
                break
        if type(change) == str:
            raise (parameter, change)
        data = {"img": str(img_docker_dir / img_name), "parameter": parameter, "change": change, "out": str(out_docker_dir / parameter / str(change) / img_name)}
        pipe.rpush("NIAA_img_q", json.dumps(data))
    pipe.execute()
else:
    for i, image in enumerate(list(img_dir.iterdir())):
        if i % 1000 == 0:
            print(f"{i} images done")
        for parameter in parameter_range:
            for change in parameter_range[parameter]["range"]:
                if math.isclose(change, parameter_range[parameter]["default"]):
                    continue
                data = {"img": str(img_docker_dir / image.name), "parameter": parameter, "change": change, "out": str(out_docker_dir / parameter / str(change) / image.name)}

                if mode == "all":
                    pipe.rpush("NIAA_img_q", json.dumps(data))
                elif mode == "png" and image.suffix.lower() == ".png":
                    pipe.rpush("NIAA_img_q", json.dumps(data))
                    continue
        pipe.execute()
