import sys

import pandas as pd

sys.path.insert(0, ".")
from SSMTIA.utils import mapping

chunks = []

file = "analysis/not_uploaded/MPADA_test_scores.csv"

df = pd.read_csv(file, sep=";")
df["low_quality"] = df["scores"].apply(lambda row: eval(row)[0])
df["high_quality"] = df["scores"].apply(lambda row: eval(row)[1])

del df["scores"]

df.to_csv("analysis/not_uploaded/parsed/MPADA_test_scores.csv", index=False)
