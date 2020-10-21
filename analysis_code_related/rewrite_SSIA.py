import sys

import pandas as pd

sys.path.insert(0, ".")
from SSMTIA.utils import mapping

chunks = []

file = "analysis/not_uploaded/SSIA_mobilenet_test_scores.csv"

for chunk in pd.read_csv(file, sep=";", chunksize=10000):
    chunk["scores"] = chunk["scores"].apply(eval)

    chunk["styles_score"] = chunk["scores"].apply(lambda row: row["styles_score"][0]).astype("float16")
    chunk["technical_score"] = chunk["scores"].apply(lambda row: row["technical_score"][0]).astype("float16")
    chunk["composition_score"] = chunk["scores"].apply(lambda row: row["composition_score"][0]).astype("float16")

    chunk = chunk.drop(columns=["scores"])
    chunks.append(chunk)
df = pd.concat(chunks)
# del chunks
# df = df.melt(id_vars=["img", "parameter", "change", "styles_score", "technical_score", "composition_score"], var_name="pred_change", value_name="pred_change_degree")

df.to_csv("analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores.csv", index=False)
