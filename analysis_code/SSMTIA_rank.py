import pandas as pd
from scipy import stats
import sys
import math

sys.path.insert(0, ".")
from SSMTIA.utils import mapping, parameter_range

out_file = open("/workspace/analysis/SSMTIA/spearman.txt", "w")

df = pd.read_csv("/workspace/analysis/not_uploaded/parsed/SSMTIA_mobilenet_test_scores.csv")

df["score"] = (df["styles_score"] + df["technical_score"] + df["composition_score"]) / 3
df = df[["img", "parameter", "change", "score"]]


def calculate_spearman(distortion: str, polarity: str, img_names=df["img"].unique()):
    corr_l = []
    p_l = []
    original_df = df[df["parameter"] == "original"]
    if distortion in parameter_range:
        original_df["change"] = parameter_range[distortion]["default"]

    parameter_df = df[df["parameter"] == distortion]
    corr_df = pd.concat([parameter_df, original_df])[["img", "parameter", "change", "score"]]

    for i, f in enumerate(img_names):
        if i % 1000 == 0:
            print(i)
        corr_df_img = corr_df[corr_df["img"] == f]

        if distortion in parameter_range:
            default = parameter_range[distortion]["default"]
        else:
            default = 0

        if polarity == "pos":
            corr_df_img = corr_df_img[corr_df_img["change"] >= default]
        else:
            corr_df_img = corr_df_img[corr_df_img["change"] <= default]

        corr_df_img["change"] = corr_df_img["change"].apply(lambda x: abs((x) - (default)))

        c, p = stats.spearmanr(corr_df_img["score"], corr_df_img["change"])
        if math.isnan(c) or math.isnan(p):
            continue
        corr_l.append(c)
        p_l.append(p)
    return sum(corr_l) / len(corr_l), sum(p_l) / len(p_l)


for distortion in list(mapping["styles"].keys()) + list(mapping["technical"].keys()) + list(mapping["composition"].keys()):
    print(distortion)
    try:
        out_file.write(f"{distortion},pos,{calculate_spearman(distortion=distortion, polarity='pos')}\n")
    except:
        out_file.write(f"{distortion},pos, didnt work")
    out_file.flush()
    try:
        out_file.write(f"{distortion},neg,{calculate_spearman(distortion=distortion, polarity='neg')}\n")
    except:
        out_file.write(f"{distortion},neg, didnt work")
    out_file.flush()

out_file.close()
