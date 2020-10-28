import sys
from pathlib import Path

import torch
import torchvision.models as models
import logging

sys.path[0] = "/workspace"
from relatedWorks.NIMA import NIMA
from relatedWorks.datasets import AVA
#from IA.utils import mapping

test_file = "/workspace/dataset_processing/test_set.txt"
model_path = "/scratch/pretrained_new.pth"
out_file = "/workspace/analysis/not_uploaded/NIMA_test_scores.csv"

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

model = NIMA(models.vgg16(pretrained=False))

model.load_state_dict(torch.load(model_path))
logging.info("successfully loaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

logging.info("creating dataloader")
#dataset = SSPexels(file_list_path=test_file, mapping=mapping)
dataset = AVA()
batch_loader = torch.utils.data.DataLoader(dataset, batch_size=30, drop_last=False, num_workers=8)

out_file = open(out_file, "w")
out_file.write("img;score\n")

logging.info("testing")
for i, data in enumerate(batch_loader):
    logging.info(f"{i}/{len(batch_loader)}")

    img = data["img"].to(device)
    with torch.no_grad():
        out = model(img)
    for p, s in zip(data["file_name"], out):
        out_file.write(f"{p};{s.tolist()}\n")

out_file.close()
