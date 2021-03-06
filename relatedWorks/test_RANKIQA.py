import sys
from pathlib import Path

import torch
import caffemodel2pytorch
import logging

sys.path[0] = "/workspace"
from relatedWorks.datasets import SSPexels
from SSMTIA.utils import mapping

test_file = "/workspace/dataset_processing/test_set.txt"
out_file = "/workspace/analysis/not_uploaded/RANKIQA_test_scores.csv"

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

model = caffemodel2pytorch.Net(prototxt="/relatedNets/RankIQA/src/FT/tid2013/deploy_vgg.prototxt", weights="/relatedNets/RankIQA/models/RankIQA_models/FT_tid2013.caffemodel", caffe_proto="https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto")

logging.info("successfully loaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

logging.info("creating dataloader")
dataset = SSPexels(file_list_path=test_file, mapping=mapping, normalize=False)
batch_loader = torch.utils.data.DataLoader(dataset, batch_size=30, drop_last=False, num_workers=8)

out_file = open(out_file, "w")
out_file.write("img;parameter;change;scores\n")

logging.info("testing")
for i, data in enumerate(batch_loader):
    logging.info(f"{i}/{len(batch_loader)}")
    for key in data.keys():
        if key == "file_name":
            continue

        img = data[key].to(device)
        with torch.no_grad():
            out = model(img)["fc8"]
        for p, s in zip(data["file_name"], out):
            if key == "original":
                key = "original;0"
            out_file.write(f"{p};{key};{s.tolist()[0]}\n")

out_file.close()
