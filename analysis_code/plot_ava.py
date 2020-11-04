import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
import math

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

gt = pd.read_csv("/scratch/stud/pfister/NIAA/AVA/AVA.txt", sep=" ").drop(columns=["Unnamed: 0", "semanticTagID1", "semanticTagID2", "challengeID"])
ours = pd.read_csv("/home/stud/pfister/eclipse-workspace/NIAA/analysis/not_uploaded/IA/AVA/.scratch.ckpts.IA.pexels.scores-one.change_regress.epoch-9.pth.txt")

# ours["score"] = (ours["styles_score"] + ours["technical_score"]) / 2

gt["votes"] = gt.apply(lambda row: sum(list(row)[1:]), axis=1)
gt["gt_score"] = gt.apply(lambda row: sum([val * (i + 1) for i, val in enumerate(list(row)[1:-1])]), axis=1)
gt["gt_score"] = gt.apply(lambda row: row.gt_score / row.votes, axis=1)
gt["gt_quality"] = gt["gt_score"].apply(lambda row: 1 if row > 5 else 0)

print(gt)
print(ours)

gt = gt[["img", "gt_score", "gt_quality"]]

gt["img"] = gt["img"].astype(int)

ours["img"] = ours["img"].apply(lambda row: int(Path(row).stem))
ours["score"] = ours["score"].apply(lambda row: eval(row)[0])

df = gt.merge(ours, left_on="img", right_on="img")
df = df.dropna()


df = pd.DataFrame(df.to_records()).drop(columns="index")

df = df[df["img"].isin(pd.read_csv("/scratch/stud/pfister/NIAA/AVA/test_labels.csv")["file"])]

print(df)

print(stats.spearmanr(df["gt_score"], df["score"]))

# print(stats.spearmanr(df["gt_score"], df["styles_score"]))
# print(stats.spearmanr(df["gt_score"], df["technical_score"]))
# print(stats.spearmanr(df["gt_score"], df["composition_score"]))


for cutoff in np.arange(0, 1, 0.05):
    df["quality"] = df["score"].apply(lambda row: 1 if row > cutoff else 0)
    print(cutoff, accuracy_score(df["gt_quality"], df["quality"]))

# sns.histplot(data=df, x="gt_score", y="score")
sns.pairplot(data=df.drop(columns=["img", "gt_quality", "quality"]), kind="hist", corner=True)
plt.savefig("dist.png")