import argparse
import json
import math
import sys
from pathlib import Path

import redis

sys.path[0] = "/workspace"
from edit_image import parameter_range

orig_dir: Path = Path("/scratch/stud/pfister/NIAA/pexels/images")
edited_dir: Path = Path("/scratch/stud/pfister/NIAA/pexels/edited_images")

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str)
parser.add_argument("--file_list", type=str)
parser.add_argument("--parameters", type=str, nargs="+")
parser.add_argument("--compare_opposite_polarity", action="store_true")
parser.add_argument("--original_present", action="store_true")

args = parser.parse_args()

if args.mode == "train":
    db = 0
elif args.mode == "val":
    db = 1
elif args.mode == "test":
    db = 2
else:
    raise NotImplementedError("?")

r = redis.Redis(host="redisdataset", db=db)
r.flushdb()
pipeline = r.pipeline()

edits = []
i = 0

files = Path(args.file_list).read_text()
files = files.split("\n")
files = [val.strip() for val in files]

for img in files:
    print(img)
    for parameter in args.parameters:
        if args.original_present:
            for change in parameter_range[parameter]["range"]:
                if math.isclose(change, parameter_range[parameter]["default"]):
                    continue
                relDist = abs((parameter_range[parameter]["default"]) - (change))
                relDist = 0 if math.isclose(relDist, 0) else relDist
                relDist = round(relDist, 2)

                pipeline.set(i, json.dumps({"img1": str(orig_dir / img), "img2": str(edited_dir / parameter / str(change) / img), "parameter": parameter, "changes1": parameter_range[parameter]["default"], "changes2": change, "relChanges1": 0, "relChanges2": relDist}))
                i += 1

        else:
            for lchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
                for rchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
                    if math.isclose(lchange, rchange):  # don't compare 0.5 to 0.5 for example
                        continue
                    if rchange < lchange:  # only compare 0.4 to 0.5 but not 0.5 to 0.4
                        continue
                    if not args.compare_opposite_polarity:
                        if lchange < parameter_range[parameter]["default"] and rchange > parameter_range[parameter]["default"] or lchange > parameter_range[parameter]["default"] and rchange < parameter_range[parameter]["default"]:
                            continue

                    lRelDist = abs((parameter_range[parameter]["default"]) - (lchange))
                    rRelDist = abs((parameter_range[parameter]["default"]) - (rchange))
                    lRelDist = round(lRelDist, 2)
                    rRelDist = round(rRelDist, 2)

                    if lRelDist > rRelDist or math.isclose(lRelDist, rRelDist):  # smaller change always has to be img1/imgl, as the Distance Loss assumes the "more original" image is the first
                        continue

                    if math.isclose(lchange, parameter_range[parameter]["default"]):
                        imgl = str(orig_dir / img)
                    else:
                        imgl = str(edited_dir / parameter / str(lchange) / img)
                    if math.isclose(rchange, parameter_range[parameter]["default"]):
                        imgr = str(orig_dir / img)
                    else:
                        imgr = str(edited_dir / parameter / str(rchange) / img)

                    pipeline.set(i, json.dumps({"img1": imgl, "img2": imgr, "parameter": parameter, "changes1": lchange, "changes2": rchange, "relChanges1": lRelDist, "relChanges2": rRelDist}))
                    i += 1
        pipeline.execute()

pipeline.execute()
