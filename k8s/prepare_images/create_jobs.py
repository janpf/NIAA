import json
import math
import sys
from pathlib import Path

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

del parameter_range["lcontrast"]
for i, image in enumerate(list(img_dir.iterdir())):
    if i % 1000 == 0:
        print(f"{i} images done")
    for parameter in parameter_range:
        for change in parameter_range[parameter]["range"]:
            if math.isclose(change, parameter_range[parameter]["default"]):
                continue
            # print({"img": str(img_docker_dir / image.name), "parameter": parameter, "change": change, "out": str(out_docker_dir / parameter / str(change) / image.name)})
            pipe.rpush("NIAA_img_q", json.dumps({"img": str(img_docker_dir / image.name), "parameter": parameter, "change": change, "out": str(out_docker_dir / parameter / str(change) / image.name)}))
    pipe.execute()
