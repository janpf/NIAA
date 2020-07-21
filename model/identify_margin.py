import sys
from pathlib import Path

import math
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

sys.path[0] = "/workspace"

from model.NIAA import NIAA
from edit_image import parameter_range


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

orig_dir: Path = Path("/scratch/pexels/images")
edited_dir: Path = Path("/scratch/pexels/edited_images")

# fmt: off
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])
# fmt: on

base_model = models.vgg16(pretrained=True)
model = NIAA(base_model).to(device)

for img in orig_dir.iterdir():
    for parameter in parameter_range.keys():
        for lchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
            for rchange in parameter_range[parameter]["range"]:  # iterating over all possible changes
                if math.isclose(lchange, rchange):  # don't compare 0.5 to 0.5 for example
                    continue
                if rchange < lchange:  # only compare 0.4 to 0.5 but not 0.5 to 0.4
                    continue
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

                img1 = transform(Image.open(imgl).convert("RGB")).unsqueeze(dim=0).to(device)
                img2 = transform(Image.open(imgr).convert("RGB")).unsqueeze(dim=0).to(device)

                with torch.no_grad():
                    out1, out2 = model(img1, img2, "siamese")

                out1 = out1.data[0]
                out2 = out2.data[0]

                print(f"{out1}, {out2}, {parameter}, {lchange}, {rchange}, {lRelDist}, {rRelDist}")
