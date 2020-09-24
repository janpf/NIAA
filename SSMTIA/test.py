import sys
import logging


import torch


sys.path[0] = "/workspace"
from SSMTIA.dataset import SSPexels
from SSMTIA.SSMTIA import SSMTIA
from SSMTIA.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)


test_file = "/workspace/dataset_processing/test_set.txt"
model_path = "/scratch/ckpts/SSMTIA-CP/pexels/mobilenet/epoch-10.pth"
out_file = "/workspace/analysis/not_uploaded/SSMTIA_test_scores.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")
ssmtia = SSMTIA("mobilenet", mapping, pretrained=False).to(device)
ssmtia.load_state_dict(torch.load(model_path))
logging.info("using half precision")
ssmtia.half()

logging.info("creating datasets")
# datasets
SSPexels_test = SSPexels(file_list_path=test_file, mapping=mapping, return_file_name=True)
Pexels_test = torch.utils.data.DataLoader(SSPexels_test, batch_size=25, drop_last=False, num_workers=16)
logging.info("datasets created")

exit()
# FIXME model.eval

out_file = open(out_file, "w")
out_file.write("img;parameter;change;scores\n")


logging.info("testing")

for i, data in enumerate(Pexels_test):
    logging.info(f"{i}/{len(Pexels_test)}")

    for key in data.keys():
        if key == "file_name":
            continue

        img = data[key].to(device)
        with torch.no_grad():
            out = ssmtia(img)
        result_dicts = [dict() for _ in range(len(data["file_name"]))]

        for k in out.keys():
            for i in range(len(data["file_name"])):
                result_dicts[i][k] = out[k].tolist()[i]
        for p, s in zip(data["file_name"], result_dicts):
            if key == "original":
                key = "original;0"
            out_file.write(f"{p};{key};{s}\n")

out_file.close()
