import sys
from pathlib import Path

import math
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

sys.path[0] = "/workspace"

from model.datasets import PexelsRedis
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

Pexels_train = PexelsRedis(mode="train", transforms=transform)
Pexels_train_loader = torch.utils.data.DataLoader(Pexels_train, batch_size=1, shuffle=True, num_workers=50)

for i, data in enumerate(Pexels_train_loader):
    img1 = data["img1"].to(device)
    img2 = data["img2"].to(device)

    with torch.no_grad():
        out1, out2 = model(img1, img2, "siamese")
        out1 = out1.data[0]
        out2 = out2.data[0]

        print(f"{out1}, {out2}, {data['parameter'][0]}, {data['changes1'].data[0]}, {data['changes2'].data[0]}, {data['relChanges1'].data[0]}, {data['relChanges2'].data[0]}")
