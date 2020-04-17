import math
import collections
import json
from pathlib import Path
import pandas as pd
import random
import sys
import time

sys.path.insert(0, ".")
from edit_image import parameter_range, random_parameters

submission_log = "/home/stud/pfister/random.log"
with open(submission_log, "w") as f:
    for _ in range(100000):
        changes = random_parameters()
        data = {
            "loadTime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "submitTime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "img": "whatever.jpg",
            "parameter": changes[0],
            "leftChanges": changes[1][0],
            "rightChanges": changes[1][1],
            "chosen": random.choice(["leftImage", "rightImage", "unsure"]),
            "hashval": 123,
            "screenWidth": 123,
            "screenHeight": 123,
            "windowWidth": 123,
            "windowHeight": 123,
            "colorDepth": 123,
            "id": random.randint(0, 500),
            "count": random.randint(0, 500),
            "useragent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.5 Safari/605.1.15",
        }
        f.write(f"submit: {data}\n")
