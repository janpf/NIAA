import sys
from pathlib import Path


import pandas as pd

sys.path.insert(0, ".")
from SSMTIA.utils import mapping

chunks = []

file = "analysis/not_uploaded/SSIA_mobilenet_test_scores_TID2013.csv"

df = pd.read_csv(file, sep=";")

df["img"] = df["img"].apply(lambda row: Path(row).name.lower())


df["scores"] = df["scores"].apply(eval)

df["styles_score"] = df["scores"].apply(lambda row: row["styles_score"][0]).astype("float16")
df["technical_score"] = df["scores"].apply(lambda row: row["technical_score"][0]).astype("float16")
df["composition_score"] = df["scores"].apply(lambda row: row["composition_score"][0]).astype("float16")

df = df.drop(columns=["scores"])

df["score"] = (df["styles_score"] + df["technical_score"]) / 2

df = df.sort_values(by=["img"])
df.to_csv("analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores_TID2013.csv", index=False)

for col in ["score", "styles_score", "technical_score", "composition_score"]:
    df[col] = df[col].apply(lambda row: row * 10)
    df[col] = df[col].apply(lambda row: "{:.5f}".format(row))

df[df["img"].str.contains("_")][["score", "img"]].to_csv("analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores_TID2013_avg.csv", index=False, header=False, sep=" ")
df[df["img"].str.contains("_")][["technical_score", "img"]].to_csv("analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores_TID2013_tech.csv", index=False, header=False, sep=" ")
