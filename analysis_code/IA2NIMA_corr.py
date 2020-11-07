import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd
from scipy import stats


sys.path.insert(0, ".")
from IA.utils import mapping, parameter_range

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument("--resultfile", type=str)

config = parser.parse_args()

df = pd.read_csv(config.resultfile)

df["scores"] = df["scores"].apply(eval)
df["score"] = df["scores"].apply(lambda row: sum([row[i] * (i + 1) for i in range(len(row))]))

out_file = dict()
out_file["spearman"] = open(f"/workspace/analysis/IA2NIMA/{Path(config.resultfile).stem}_spearman.txt", "w")
out_file["pearson"] = open(f"/workspace/analysis/IA2NIMA/{Path(config.resultfile).stem}_pearson.txt", "w")

df = df[["img", "distortion", "level", "score"]]


def calculate_corr(distortion: str, polarity: str, corr: str, img_names=df["img"].unique()):
    corr_l = []
    p_l = []
    original_df = df[df["distortion"] == "original"]
    if distortion in parameter_range:
        original_df["level"] = parameter_range[distortion]["default"]

    parameter_df = df[df["distortion"] == distortion]
    corr_df = pd.concat([parameter_df, original_df])[["img", "distortion", "level", "score"]]

    if distortion in parameter_range:
        default = parameter_range[distortion]["default"]
    else:
        default = 0

    if polarity == "pos":
        corr_df = corr_df[corr_df["level"] >= default]
    else:
        corr_df = corr_df[corr_df["level"] <= default]

    corr_df["level"] = corr_df["level"].apply(lambda x: abs((x) - (default)))

    for i, f in enumerate(img_names):
        if i % 1000 == 0:
            logging.info(i)
        corr_df_img = corr_df[corr_df["img"] == f]

        if corr == "spearman":
            c, p = stats.spearmanr(corr_df_img["score"], corr_df_img["level"])
        elif corr == "pearson":
            c, p = stats.pearsonr(corr_df_img["score"], corr_df_img["level"])

        if math.isnan(c) or math.isnan(p):
            continue
        corr_l.append(c)
        p_l.append(p)
    return sum(corr_l) / len(corr_l), sum(p_l) / len(p_l)


for distortion in list(mapping["styles"].keys()) + list(mapping["technical"].keys()) + list(mapping["composition"].keys()):
    logging.info(distortion)

    for corr in ["pearson", "spearman"]:
        for polarity in ["pos", "neg"]:
            try:
                out_file[corr].write(f"{distortion},{polarity},{calculate_corr(distortion=distortion, polarity=polarity, corr=corr)}\n")
            except:
                out_file[corr].write(f"{distortion},{polarity}, didnt work\n")

            out_file[corr].flush()
