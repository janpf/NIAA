from pathlib import Path
from imageai.Prediction import ImagePrediction

import requests

print("downloading model")
r = requests.get("https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/DenseNet-BC-121-32.h5")
with open("/tmp/DenseNet-BC-121-32.h5", "wb") as f:
    f.write(r.content)

print("loading model")
orig_dir: Path = Path("/scratch/pexels/images")

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()  # https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/DenseNet-BC-121-32.h5
prediction.setModelPath("/tmp/DenseNet-BC-121-32.h5")
prediction.loadModel()

for img in orig_dir.iterdir():
    print(img.name, end="")
    print(",", end="")
    predictions, probabilities = prediction.predictImage(str(img), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(f"{eachPrediction}:{eachProbability},", end="")
    print(flush=True)
