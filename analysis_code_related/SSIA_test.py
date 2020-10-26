import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, ".")

from SSIA.utils import mapping, parameter_range


def histogram_distortion(distortion: str, score: str):
    plt.clf()
    sns.distplot(df[df["parameter"] == "original"][score], label="original")
    for change in (val for val in mapping["all_changes"] if distortion in val):
        parameter, change = change.split(";")
        sns.distplot(df[(df["parameter"] == parameter) & (df["change"] == float(change))][score], label=f"{parameter}: {change}")
    plt.legend()
    plt.savefig(f"/workspace/analysis/SSIA/mobilenet/hist_{distortion}_{score}.png")
    plt.clf()


def violin_distortion(distortion: str, score: str):
    plt.clf()
    plot_frame = df[(df["parameter"] == distortion) | (df["parameter"] == "original")]
    if distortion in parameter_range:
        plot_frame.loc[plot_frame["parameter"] == "original", "change"] = parameter_range[distortion]["default"]
    sns.violinplot(data=plot_frame, x="change", y=score, color="steelblue")  # TODO colors for scores
    plt.savefig(f"/workspace/analysis/SSIA/mobilenet/viol_{distortion}_{score}.png")
    plt.clf()


df = pd.read_csv("/workspace/analysis/not_uploaded/parsed/SSIA_mobilenet_test_scores.csv")
df["score"] = (df["styles_score"] + df["technical_score"] + df["composition_score"]) / 3


avg_scores = df[["parameter", "change", "score", "styles_score", "technical_score", "composition_score"]].groupby(["parameter", "change"]).mean().reset_index()
avg_scores.to_csv("/workspace/analysis/SSIA/avg_scores.csv", index=False)

sns.distplot(df["score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["score"], label="original")
plt.legend()
plt.savefig("/workspace/analysis/SSIA/mobilenet/original_avg.png")
plt.clf()

sns.distplot(df["styles_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["styles_score"], label="original")
plt.legend()
plt.savefig("/workspace/analysis/SSIA/mobilenet/original_styles.png")
plt.clf()

sns.distplot(df["technical_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["technical_score"], label="original")
plt.legend()
plt.savefig("/workspace/analysis/SSIA/mobilenet/original_technical.png")
plt.clf()

sns.distplot(df["composition_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["composition_score"], label="original")
plt.legend()
plt.savefig("/workspace/analysis/SSIA/mobilenet/original_composition.png")
plt.clf()

for dist_type in ["styles", "technical", "composition"]:
    for dist in mapping[dist_type].keys():
        for score in ["styles_score", "technical_score", "composition_score"]:
            print(dist)
            histogram_distortion(dist, score)
            violin_distortion(dist, score)
