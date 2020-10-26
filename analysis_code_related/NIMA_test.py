import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, ".")

from SSMTIA.utils import mapping, parameter_range


def histogram_distortion(distortion: str):
    plt.clf()
    sns.distplot(df[df["parameter"] == "original"]["score"], label="original")
    for change in (val for val in mapping["all_changes"] if distortion in val):
        parameter, change = change.split(";")
        sns.distplot(df[(df["parameter"] == parameter) & (df["change"] == float(change))]["score"], label=f"{parameter}: {change}")
    plt.legend()
    plt.savefig(f"/workspace/analysis/NIMA/hist_{distortion}.png")
    plt.clf()


def violin_distortion(distortion: str):
    plt.clf()
    plot_frame = df[(df["parameter"] == distortion) | (df["parameter"] == "original")]
    if distortion in parameter_range:
        plot_frame.loc[plot_frame["parameter"] == "original", "change"] = parameter_range[distortion]["default"]
    sns.violinplot(data=plot_frame, x="change", y="score", color="steelblue")
    plt.savefig(f"/workspace/analysis/NIMA/viol_{distortion}.png")
    plt.clf()


def violin_changes_original(distortion: str):
    original_frame = df[df["parameter"] == "original"]
    results = []
    df_index = list(original_frame.columns).index(f"{distortion}_change_strength")
    for index, row in original_frame.iterrows():
        for i, k in enumerate(mapping[distortion].keys()):
            results.append({"distortion": distortion, "change_predict": k, "change_strength": row[df_index][i]})
    results = (pd.DataFrame([val], columns=val.keys()) for val in results)
    sns.violinplot(data=pd.concat(results, ignore_index=True), x="change_predict", y="change_strength", color="steelblue")


df = pd.read_csv("/workspace/analysis/not_uploaded/NIMA_test_dist.csv", sep=";")
df["dist"] = df["dist"].apply(eval)
df["score"] = df["dist"].apply(lambda row: sum([row[i] * (i + 1) for i in range(len(row))]))

sns.distplot(df["score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["score"], label="original")
plt.legend()
plt.savefig("/workspace/analysis/NIMA/original.png")
plt.clf()

for dist_type in ["styles", "technical", "composition"]:
    for dist in mapping[dist_type].keys():
        print(dist)
        histogram_distortion(dist)
        violin_distortion(dist)
