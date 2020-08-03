from pathlib import Path
from imageai.Detection import ObjectDetection

import requests

print("downloading model")
r = requests.get("https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5")
with open("/tmp/resnet50_coco_best_v2.0.1.h5", "wb") as f:
    f.write(r.content)

print("loading model")
orig_dir: Path = Path("/scratch/pexels/images")

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()  # https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
detector.setModelPath("/tmp/resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

for img in orig_dir.iterdir():
    print(img.name, end="")
    print(",", end="")
    detections = detector.detectObjectsFromImage(input_image=str(img), minimum_percentage_probability=30)
    for obj in detections:
        print(f"{obj['name']}:{obj['percentage_probability']},", end="")
    print(flush=True)
