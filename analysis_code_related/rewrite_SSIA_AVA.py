import sys
from pathlib import Path


import pandas as pd

sys.path.insert(0, ".")
from SSMTIA.utils import mapping

chunks = []

file = "analysis/not_uploaded/SSIA_mobilenet_test_scores_AVA.csv"

df = pd.read_csv(file, sep=";")

df["img"] = df["img"].apply(lambda row: int(Path(row).stem))

df["scores"] = df["scores"].apply(eval)

df["styles_score"] = df["scores"].apply(lambda row: row["styles_score"][0]).astype("float16")
df["technical_score"] = df["scores"].apply(lambda row: row["technical_score"][0]).astype("float16")
df["composition_score"] = df["scores"].apply(lambda row: row["composition_score"][0]).astype("float16")

df = df.drop(columns=["scores"])

df["score"] = (df["styles_score"] + df["technical_score"] + df["composition_score"]) / 3
df.to_csv("analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores_AVA.csv", index=False)
