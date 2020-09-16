import sys

import streamlit as st
import torch
import torchvision
from PIL import Image
import numpy as np

sys.path.insert(0, ".")

from SSMTIA.SSMTIA import SSMTIA
from SSMTIA.utils import mapping

st.set_option("deprecation.showfileUploaderEncoding", False)


def pad_square(im: Image.Image, min_size: int = 224, fill_color=(0, 0, 0)) -> Image.Image:
    im = torchvision.transforms.Resize(224)(im)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new("RGB", (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return torchvision.transforms.Resize(224)(new_im)


st.title("Test out the model")


# loading model
model_path = "/home/codespace/workspace/NIAA/analysis/not_uploaded/epoch-10.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssmtia = SSMTIA("mobilenet", mapping, pretrained=False)
ssmtia.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# uploading image
img = st.file_uploader("upload an image")
# display image
img = pad_square(Image.open(img))


"The input image as seen by the net:"
st.image(np.array(img))

# predicting
img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
result = ssmtia(img)

for k in result.keys():
    result[k] = result[k].tolist()[0]
    if "score" in k:
        result[k] = result[k][0]
    if "strength" in k:
        changes = result[k]
        distortion = k.split("_")[0]
        result[k] = dict()

        for i, element in enumerate(changes):
            result[k][list(mapping[distortion].keys())[i]] = element


result
