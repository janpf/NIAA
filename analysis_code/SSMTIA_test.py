import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, ".")

from SSMTIA.utils import mapping


def histogram_distortion(distortion: str, score: str):
    plt.clf()
    sns.distplot(df[df["parameter"] == "original"][score], label="original")
    for change in (val for val in mapping["all_changes"] if distortion in val):
        parameter, change = change.split(";")
        sns.distplot(df[(df["parameter"] == parameter) & (df["change"] == float(change))][score], label=f"{parameter}: {change}")
    plt.legend()
    plt.savefig(f"analysis/SSMTIA/mobilenet/hist_{distortion}_{score}.png")
    plt.clf()


def violin_distortion(distortion: str, score: str):  # FIXME defaults for shadows, hightlights...
    plt.clf()
    plot_frame = df[(df["parameter"] == distortion) | (df["parameter"] == "original")]
    sns.violinplot(data=plot_frame, x="change", y=score, color="steelblue")  # TODO colors for scores
    plt.savefig(f"analysis/SSMTIA/mobilenet/viol_{distortion}_{score}.png")
    plt.clf()


chunks = []

for chunk in pd.read_csv("analysis/not_uploaded/SSMTIA_test_scores.csv", sep=";", chunksize=100000):
    chunk["scores"] = chunk["scores"].apply(eval)

    chunk["styles_score"] = chunk["scores"].apply(lambda row: row["styles_score"][0]).astype("float16")
    chunk["technical_score"] = chunk["scores"].apply(lambda row: row["technical_score"][0]).astype("float16")
    chunk["composition_score"] = chunk["scores"].apply(lambda row: row["composition_score"][0]).astype("float16")

    chunk["styles_change_strength"] = chunk["scores"].apply(lambda row: row["styles_change_strength"])
    chunk["technical_change_strength"] = chunk["scores"].apply(lambda row: row["technical_change_strength"])
    chunk["composition_change_strength"] = chunk["scores"].apply(lambda row: row["composition_change_strength"])

    chunk = chunk.drop(columns=["scores"])
    chunks.append(chunk)
df = pd.concat(chunks)
del chunks

sns.distplot(df["styles_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["styles_score"], label="original")
plt.legend()
plt.savefig("analysis/SSMTIA/mobilenet/original_styles.png")
plt.clf()

sns.distplot(df["technical_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["technical_score"], label="original")
plt.legend()
plt.savefig("analysis/SSMTIA/mobilenet/original_technical.png")
plt.clf()

sns.distplot(df["composition_score"], label="overall")
sns.distplot(df[df["parameter"] == "original"]["composition_score"], label="original")
plt.legend()
plt.savefig("analysis/SSMTIA/mobilenet/original_composition.png")
plt.clf()

for dist_type in ["styles", "technical", "composition"]:
    for dist in mapping[dist_type].keys():
        for score in ["styles_score", "technical_score", "composition_score"]:
            print(dist)
            histogram_distortion(dist, score)
            violin_distortion(dist, score)
