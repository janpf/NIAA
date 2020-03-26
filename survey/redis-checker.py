import redis
import json
from pathlib import Path

r = redis.Redis(host="localhost", port=7000)

qlen = r.llen("q")
pairslen = r.llen("pairs")
imgslen = r.hlen("imgs")

print(f"q:\t{qlen}")
print(f"pairs:\t{pairslen}")
print(f"imgslen:{imgslen}")
print()

data = r.lrange("pairs", 0, 10000)

data = [json.loads(val) for val in data]
imgsPairs = [val["img"] for val in data]

print(r.hlen("imgs"))
print(len(imgsPairs))
print(len(set(imgsPairs)))

input("passt?")

found = 0
one_not_found = 0
both_not_found = 0

for img in imgsPairs:
    left = f"{Path(img).stem}_l.jpg"
    right = f"{Path(img).stem}_r.jpg"

    left_ex = r.hexists("imgs", left)
    right_ex = r.hexists("imgs", right)

    if left_ex and right_ex:
        found += 1
    elif left_ex or right_ex:
        one_not_found += 1
    else:
        both_not_found += 1

print(f"both found:\t{found}")
print(f"one found:\t{one_not_found}")
print(f"both missing:\t{both_not_found}")
