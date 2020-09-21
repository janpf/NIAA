import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, ".")

from SSMTIA.utils import mapping


def histogram_distortion(distortion: str):
    plt.clf()
    sns.distplot(df[df["parameter"] == "original"]["score"], label="original")
    for change in (val for val in mapping["all_changes"] if distortion in val):
        parameter, change = change.split(";")
        sns.distplot(df[(df["parameter"] == parameter) & (df["change"] == float(change))]["score"], label=f"{parameter}: {change}")
    plt.legend()
    plt.savefig(f"analysis/NIMA/hist_{distortion}.png")
    plt.clf()


def violin_distortion(distortion: str):  # FIXME defaults for shadows, hightlights...
    plt.clf()
    plot_frame = df[(df["parameter"] == distortion) | (df["parameter"] == "original")]
    sns.violinplot(data=plot_frame, x="change", y="score", color="steelblue")
    plt.savefig(f"analysis/NIMA/viol_{distortion}.png")
    plt.clf()


df = pd.read_csv("analysis/not_uploaded/NIMA_test_dist.csv", sep=";")
df["dist"] = df["dist"].apply(eval)
df["score"] = df["dist"].apply(lambda row: sum([row[i] * (i + 1) for i in range(len(row))]))

sns.distplot(df["score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["score"], label="original")
plt.legend()
plt.savefig("analysis/NIMA/original.png")
plt.clf()

for dist_type in ["styles", "technical", "composition"]:
    for dist in mapping[dist_type].keys():
        print(dist)
        histogram_distortion(dist)
        violin_distortion(dist)
