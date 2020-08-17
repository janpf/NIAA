import sys
from itertools import chain
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms

sys.path[0] = "/workspace"
from model.datasets import FileList
from model.NIMA import NIMA

test_file = "/workspace/dataset_processing/test_set.txt"
model_path = "/scratch/pretrained_new.pth"
orig_imgs = "/scratch/pexels/images"
edited_imgs = "/scratch/pexels/edited_images"
out_file = "/workspace/analysis/NIMA_test_scores.csv"

with open(test_file) as f:
    test_set = f.readlines()
test_set = set([val.strip() for val in test_set])

test_imgs = [val for val in chain(Path(orig_imgs).rglob("*"), Path(edited_imgs).rglob("*")) if val.name in test_set]
print(f"{len(test_imgs)} files in testset")

# fmt: off
test_transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])
# fmt: on

model = NIMA(models.vgg16(pretrained=True))

model.load_state_dict(torch.load(model_path))
print("successfully loaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

file_dataset = FileList(test_imgs, test_transforms)
batch_loader = torch.utils.data.DataLoader(file_dataset, batch_size=256, drop_last=False, num_workers=24)

out_file = open(out_file, "w")

for i, data in enumerate(batch_loader):
    img = data["img"].to(device)
    with torch.no_grad():
        out = model(img)
    for p, s in zip(data["path"], out):
        out_file.write(f"{p}, {s.tolist()}\n")

out_file.close()
