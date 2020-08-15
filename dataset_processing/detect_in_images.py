from pathlib import Path
from random import shuffle

import redis
import requests
from imageai.Detection import ObjectDetection

print("downloading model")
r = requests.get("https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5")
with open("/tmp/resnet50_coco_best_v2.0.1.h5", "wb") as f:
    f.write(r.content)

print("loading model")
orig_dir: Path = Path("/scratch/pexels/images")
out_dir: Path = Path("/scratch/pexels/detected_images")

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()  # https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
detector.setModelPath("/tmp/resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

r = redis.Redis(host="redisdataset", db=15)

imgs = list(orig_dir.iterdir())
shuffle(imgs)

for img in imgs:
    print(img.name, flush=True)
    if r.exists(img.name):
        continue
    return_string = f"{img.name},"
    detections = detector.detectObjectsFromImage(input_image=str(img), output_image_path=str(out_dir / img.name), minimum_percentage_probability=50)
    for obj in detections:
        return_string += f"{obj['name']}:{obj['percentage_probability']}:{obj['box_points']},"
    r.set(img.name, return_string)
