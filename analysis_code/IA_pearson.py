import pandas as pd
from scipy import stats
import sys
import math

sys.path.insert(0, ".")
from IA.utils import mapping, parameter_range

out_file = open("/workspace/analysis/IA/pearson.txt", "w")

df = pd.read_csv("/workspace/analysis/not_uploaded/IA/.scratch.ckpts.IA.pexels.scores-one.change_regress.epoch-9.pth.txt")

# df["score"] = (df["styles_score"] + df["technical_score"] + df["composition_score"]) / 3
df = df[["img", "distortion", "level", "score"]]


def calculate_pearson(distortion: str, polarity: str, img_names=df["img"].unique()):
    corr_l = []
    p_l = []
    original_df = df[df["distortion"] == "original"]
    if distortion in parameter_range:
        original_df["level"] = parameter_range[distortion]["default"]

    parameter_df = df[df["distortion"] == distortion]
    corr_df = pd.concat([parameter_df, original_df])[["img", "distortion", "level", "score"]]

    for i, f in enumerate(img_names):
        if i % 1000 == 0:
            print(i)
        corr_df_img = corr_df[corr_df["img"] == f]
        corr_df_img["score"] = corr_df_img["score"].apply(lambda row: eval(row)[0])

        if distortion in parameter_range:
            default = parameter_range[distortion]["default"]
        else:
            default = 0

        if polarity == "pos":
            corr_df_img = corr_df_img[corr_df_img["level"] >= default]
        else:
            corr_df_img = corr_df_img[corr_df_img["level"] <= default]

        corr_df_img["level"] = corr_df_img["level"].apply(lambda x: abs((x) - (default)))

        c, p = stats.pearsonr(corr_df_img["score"], corr_df_img["level"])
        if math.isnan(c) or math.isnan(p):
            continue
        corr_l.append(c)
        p_l.append(p)
    return sum(corr_l) / len(corr_l), sum(p_l) / len(p_l)


for distortion in list(mapping["styles"].keys()) + list(mapping["technical"].keys()) + list(mapping["composition"].keys()):
    print(distortion)
    try:
        out_file.write(f"{distortion},pos,{calculate_pearson(distortion=distortion, polarity='pos')}\n")
    except:
        out_file.write(f"{distortion},pos, didnt work\n")
    out_file.flush()
    try:
        out_file.write(f"{distortion},neg,{calculate_pearson(distortion=distortion, polarity='neg')}\n")
    except:
        out_file.write(f"{distortion},neg, didnt work\n")
    out_file.flush()

out_file.close()
