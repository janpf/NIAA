import sys
from pathlib import Path

import torch
import torchvision.models as models
import logging

sys.path[0] = "/workspace"
from NIMA.NIMA import NIMA
from NIMA.datasets import SSPexels
from SSMTIA.utils import mapping

test_file = "/workspace/dataset_processing/test_set.txt"
model_path = "/scratch/pretrained_new.pth"
out_file = "/workspace/analysis/not_uploaded/NIMA_test_scores.csv"

model = NIMA(models.vgg16(pretrained=False))

model.load_state_dict(torch.load(model_path))
logging.info("successfully loaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

logging.info("creating dataloader")
dataset = SSPexels(file_list_path=test_file, mapping=mapping)
batch_loader = torch.utils.data.DataLoader(dataset, batch_size=30, drop_last=False, num_workers=8)

out_file = open(out_file, "w")
out_file.write("img; parameter; change; scores\n")

logging.info("testing")
for i, data in enumerate(batch_loader):
    logging.info(f"{i}/{len(batch_loader)}")
    for key in data.keys():
        if key == "file_name":
            continue

        img = data[key].to(device)
        with torch.no_grad():
            out = model(img)
        for p, s in zip(data["file_name"], out):
            if key == "original":
                key = "original;0"
            out_file.write(f"{p};{key};{s.tolist()}\n")

out_file.close()
