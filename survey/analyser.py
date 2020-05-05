import collections
import json
import math
import sys
from pathlib import Path

import httpagentparser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import redis
import seaborn as sns
from scipy.stats import binom_test, linregress, pearsonr, spearmanr, wilcoxon

sys.path.insert(0, ".")
from edit_image import parameter_range

sns.set(style="whitegrid")

submission_log = Path.home() / "eclipse-workspace" / "NIAA" / "survey" / "survey.csv"  # type: Path
submission_log = Path("/scratch") / "stud" / "pfister" / "NIAA" / "pexels" / "logs" / "survey_NIMA.csv"  # type: Path
plot_dir = Path.home() / "eclipse-workspace" / "NIAA" / "analysis" / "survey"  # type: Path

plot_dir.mkdir(parents=True, exist_ok=True)

try:
    r = redis.Redis(host="localhost", port=7000)
    memdata = r.info("memory")
    print(f"redis: used mem/peak: {memdata['used_memory_human']}/{memdata['used_memory_peak_human']} used rss: {memdata['used_memory_rss_human']}")

    print("preprocessing:")
    print(f"\tqueued: {r.llen('q')}")
    print(f"\tprepared: {r.llen('pairs')}")
    if r.hlen("imgs") / 2 != r.llen("pairs"):
        print(f"\timgs:{r.hlen('imgs')}")
    print("---")
    print()
except:
    print("no redis connection available => 'kctl port-forward svc/redis 7000:6379'")

sub_df = pd.read_csv(submission_log, parse_dates=["loadTime", "submitTime"])  # type: pd.DataFrame
# data reading done

print()
print(f"{sub_df.hashval.count()} images compared in {sub_df.groupby('userid').userid.count().count()} sessions")
print(sub_df.groupby("chosen").chosen.count().to_string())
print("---")
print()

# sub_df["chosen"] = sub_df.apply(lambda row: "leftImage" if row.leftNIMA > row.rightNIMA else "rightImage", axis=1)  #  if in NIMA is evaluated

sub_df["default"] = sub_df.apply(lambda row: parameter_range[row.parameter]["default"], axis=1)
sub_df["leftChanges"] = sub_df.apply(lambda row: float(row.leftChanges), axis=1)
sub_df["rightChanges"] = sub_df.apply(lambda row: float(row.rightChanges), axis=1)
sub_df["leftChanges"] = sub_df.apply(lambda row: 0 if math.isclose(row.leftChanges, 0) else row.leftChanges, axis=1)
sub_df["rightChanges"] = sub_df.apply(lambda row: 0 if math.isclose(row.rightChanges, 0) else row.rightChanges, axis=1)
sub_df["leftChanges"] = sub_df.apply(lambda row: row.default if math.isclose(row.leftChanges, row.default) else row.leftChanges, axis=1)
sub_df["rightChanges"] = sub_df.apply(lambda row: row.default if math.isclose(row.rightChanges, row.default) else row.rightChanges, axis=1)
sub_df["leftChanges"] = sub_df.apply(lambda row: round(row.leftChanges, 2), axis=1)
sub_df["rightChanges"] = sub_df.apply(lambda row: round(row.rightChanges, 2), axis=1)
sub_df["bothSame"] = sub_df.apply(lambda row: math.isclose(row.leftChanges, row.rightChanges), axis=1)

sub_df["lRelDistDefault"] = sub_df.apply(lambda row: abs((row.default) - (row.leftChanges)), axis=1)
sub_df["rRelDistDefault"] = sub_df.apply(lambda row: abs((row.default) - (row.rightChanges)), axis=1)
sub_df["lRelDistDefault"] = sub_df.apply(lambda row: 0 if math.isclose(row.lRelDistDefault, 0) else row.lRelDistDefault, axis=1)
sub_df["rRelDistDefault"] = sub_df.apply(lambda row: 0 if math.isclose(row.rRelDistDefault, 0) else row.rRelDistDefault, axis=1)
sub_df["lRelDistDefault"] = sub_df.apply(lambda row: round(row.lRelDistDefault, 2), axis=1)
sub_df["rRelDistDefault"] = sub_df.apply(lambda row: round(row.rRelDistDefault, 2), axis=1)
sub_df["smallChange"] = sub_df.apply(lambda row: row.leftChanges if row.lRelDistDefault < row.rRelDistDefault else row.rightChanges, axis=1)
sub_df["largeChange"] = sub_df.apply(lambda row: row.rightChanges if row.lRelDistDefault < row.rRelDistDefault else row.leftChanges, axis=1)
sub_df["smallRelDistDefault"] = sub_df.apply(lambda row: min(row.lRelDistDefault, row.rRelDistDefault), axis=1)
sub_df["largeRelDistDefault"] = sub_df.apply(lambda row: max(row.lRelDistDefault, row.rRelDistDefault), axis=1)
sub_df["smallLargeRelDistDefault"] = sub_df.apply(lambda row: abs((row.smallRelDistDefault) - (row.largeRelDistDefault)), axis=1)
sub_df["smallLargeRelDistDefault"] = sub_df.apply(lambda row: 0 if math.isclose(row.smallLargeRelDistDefault, 0) else row.smallLargeRelDistDefault, axis=1)
sub_df["smallLargeRelDistDefault"] = sub_df.apply(lambda row: round(row.smallLargeRelDistDefault, 2), axis=1)
sub_df["changeSign"] = sub_df.apply(lambda row: 0 if math.isclose(row.largeChange - row.default, 0) else row.largeChange - row.default, axis=1)
sub_df["changeSign"] = sub_df.apply(lambda row: np.sign(row.changeSign), axis=1)

sub_df["smallChangeIsOriginal"] = sub_df.apply(lambda row: math.isclose(row.smallChange, row.default), axis=1)

sub_df["smallerChosen"] = sub_df.apply(lambda row: not row.bothSame and ((math.isclose(row.smallChange, row.leftChanges) and row.chosen == "leftImage") or math.isclose(row.smallChange, row.rightChanges) and row.chosen == "rightImage"), axis=1)
sub_df["largerChosen"] = sub_df.apply(lambda row: not row.bothSame and ((math.isclose(row.largeChange, row.leftChanges) and row.chosen == "leftImage") or math.isclose(row.largeChange, row.rightChanges) and row.chosen == "rightImage"), axis=1)

if "leftNIMA" in sub_df.columns and "rightNIMA" in sub_df.columns:
    sub_df["chosenNIMA"] = sub_df.apply(lambda row: "leftImage" if row.leftNIMA > row.rightNIMA else "rightImage", axis=1)
    sub_df["sameNIMA"] = sub_df.apply(lambda row: row.chosen == row.chosenNIMA, axis=1)

clean_df = sub_df[sub_df.chosen != "error"]

nestedDict = lambda: collections.defaultdict(nestedDict)  # infinitely deep dict
analyzeDict = nestedDict()
for key in parameter_range.keys():
    analyzeDict[key]["overall"] = len(clean_df[clean_df.parameter == key])
    analyzeDict[key]["unsure_eq"] = len(clean_df[(clean_df.parameter == key) & (clean_df.bothSame == True) & (clean_df.chosen == "unsure")])
    analyzeDict[key]["not_unsure_eq"] = len(clean_df[(clean_df.parameter == key) & (clean_df.bothSame == True) & (clean_df.chosen != "unsure")])
    analyzeDict[key]["unsure_not_eq"] = len(clean_df[(clean_df.parameter == key) & (clean_df.bothSame == False) & (clean_df.chosen == "unsure")])

    tmp = clean_df[(clean_df.parameter == key) & (clean_df.bothSame == False) & (clean_df.chosen != "unsure")]
    analyzeDict[key]["smallerChosen"] = len(tmp[(tmp.smallerChosen == True)])
    analyzeDict[key]["smallerChosenOrigPresent"] = len(tmp[(tmp.smallerChosen == True) & (tmp.smallChangeIsOriginal == True)])
    analyzeDict[key]["largerChosen"] = len(tmp[(tmp.largerChosen == True)])
    analyzeDict[key]["largerChosenOrigPresent"] = len(tmp[(tmp.largerChosen == True) & (tmp.smallChangeIsOriginal == True)])

    d1 = tmp["leftChanges"].value_counts().to_dict()
    d2 = tmp["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChanges"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChanges"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.smallChangeIsOriginal == True)]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.smallChangeIsOriginal == True)]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChangesOrigPresent"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.smallChangeIsOriginal == True) & (tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.smallChangeIsOriginal == True) & (tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChangesOrigPresent"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}

    d1 = tmp[tmp.changeSign > 0]["leftChanges"].value_counts().to_dict()
    d2 = tmp[tmp.changeSign > 0]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChangesPos"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign > 0) & (tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign > 0) & (tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChangesPos"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign > 0) & (tmp.smallChangeIsOriginal == True)]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign > 0) & (tmp.smallChangeIsOriginal == True)]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChangesOrigPresentPos"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign > 0) & (tmp.smallChangeIsOriginal == True) & (tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign > 0) & (tmp.smallChangeIsOriginal == True) & (tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChangesOrigPresentPos"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}

    d1 = tmp[tmp.changeSign < 0]["leftChanges"].value_counts().to_dict()
    d2 = tmp[tmp.changeSign < 0]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChangesNeg"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign < 0) & (tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign < 0) & (tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChangesNeg"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign < 0) & (tmp.smallChangeIsOriginal == True)]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign < 0) & (tmp.smallChangeIsOriginal == True)]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["occuredChangesOrigPresentNeg"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}
    d1 = tmp[(tmp.changeSign < 0) & (tmp.smallChangeIsOriginal == True) & (tmp.chosen == "leftImage")]["leftChanges"].value_counts().to_dict()
    d2 = tmp[(tmp.changeSign < 0) & (tmp.smallChangeIsOriginal == True) & (tmp.chosen == "rightImage")]["rightChanges"].value_counts().to_dict()
    analyzeDict[key]["chosenChangesOrigPresentNeg"] = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()).union(set(d2.keys()))}

    analyzeDict[key]["occuredRelChangesPos"] = tmp[tmp.changeSign > 0]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["smallerChosenRelChangesPos"] = tmp[(tmp.changeSign > 0) & (tmp.smallerChosen == True)]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["occuredRelChangesOrigPresentPos"] = tmp[(tmp.changeSign > 0) & (tmp.smallChangeIsOriginal == True)]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["smallerChosenRelChangesOrigPresentPos"] = tmp[(tmp.changeSign > 0) & (tmp.smallerChosen == True) & (tmp.smallChangeIsOriginal == True)]["smallLargeRelDistDefault"].value_counts().to_dict()

    analyzeDict[key]["occuredRelChangesNeg"] = tmp[tmp.changeSign < 0]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["smallerChosenRelChangesNeg"] = tmp[(tmp.changeSign < 0) & (tmp.smallerChosen == True)]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["occuredRelChangesOrigPresentNeg"] = tmp[(tmp.changeSign < 0) & (tmp.smallChangeIsOriginal == True)]["smallLargeRelDistDefault"].value_counts().to_dict()
    analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"] = tmp[(tmp.changeSign < 0) & (tmp.smallerChosen == True) & (tmp.smallChangeIsOriginal == True)]["smallLargeRelDistDefault"].value_counts().to_dict()

    chosenVS = tmp.groupby(["smallerChosen", "smallChange", "largeChange"])[["smallerChosen"]].size().to_frame("size").reset_index()
    allComparisons = tmp.groupby(["smallChange", "largeChange"])[["smallChange"]].size().to_frame("size").reset_index()
    for _, row in chosenVS[chosenVS.smallerChosen == True].iterrows():
        if len(allComparisons[(allComparisons.smallChange == row["smallChange"]) & (allComparisons.largeChange == row["largeChange"])]) > 1:
            raise KeyError("what?")
        analyzeDict[key]["asSmallerChosenVS"][row["smallChange"]][row["largeChange"]] = row["size"] / allComparisons[(allComparisons.smallChange == row["smallChange"]) & (allComparisons.largeChange == row["largeChange"])].iloc[0]["size"]

    chosenVS = tmp.groupby(["largerChosen", "smallChange", "largeChange"])[["largerChosen"]].size().to_frame("size").reset_index()
    allComparisons = tmp.groupby(["smallChange", "largeChange"])[["largeChange"]].size().to_frame("size").reset_index()
    for _, row in chosenVS[chosenVS.largerChosen == True].iterrows():
        if len(allComparisons[(allComparisons.smallChange == row["smallChange"]) & (allComparisons.largeChange == row["largeChange"])]) > 1:
            raise KeyError("what?")
        analyzeDict[key]["asLargerChosenVS"][row["largeChange"]][row["smallChange"]] = row["size"] / allComparisons[(allComparisons.smallChange == row["smallChange"]) & (allComparisons.largeChange == row["largeChange"])].iloc[0]["size"]

    if "leftNIMA" in sub_df.columns and "rightNIMA" in sub_df.columns:
        analyzeDict[key]["sameNIMA"] = tmp["sameNIMA"].value_counts().to_dict()

    for corr in [
        ("smallerChosenRelChangesPos", "occuredRelChangesPos"),
        ("smallerChosenRelChangesOrigPresentPos", "occuredRelChangesOrigPresentPos"),
        ("smallerChosenRelChangesNeg", "occuredRelChangesNeg"),
        ("smallerChosenRelChangesOrigPresentNeg", "occuredRelChangesOrigPresentNeg"),
        ("chosenChangesPos", "occuredChangesPos"),
        ("chosenChangesOrigPresentPos", "occuredChangesOrigPresentPos"),
        ("chosenChangesNeg", "occuredChangesNeg"),
        ("chosenChangesOrigPresentNeg", "occuredChangesOrigPresentNeg"),
    ]:
        if "rel" in corr[0].lower():  # TODO implement
            pass
        elif not "rel" in corr[0].lower():
            if "pos" in corr[0].lower():
                for change in [val for val in parameter_range[key]["range"] if val >= parameter_range[key]["default"]]:
                    if not change in analyzeDict[key][corr[0]]:
                        analyzeDict[key][corr[0]][change] = 0
                    if not change in analyzeDict[key][corr[1]]:
                        raise KeyError(f"that's weird: {key} {analyzeDict[key][corr[1]].keys()} {change}")
            elif "neg" in corr[0].lower():
                if len(analyzeDict[key][corr[0]]) > 1:  # lcontrast and vibrance
                    for change in [val for val in parameter_range[key]["range"] if val <= parameter_range[key]["default"]]:
                        if not change in analyzeDict[key][corr[0]]:
                            analyzeDict[key][corr[0]][change] = 0
                        if not change in analyzeDict[key][corr[1]]:
                            raise KeyError(f"that's weird: {key} {analyzeDict[key][corr[1]].keys()} {change}")
            else:
                raise NotImplementedError("you forgot what you thought wasn't necessary")
        else:
            raise "what?"

        for val in analyzeDict[key][corr[0]].keys():
            analyzeDict[key][corr[0]][val] /= analyzeDict[key][corr[1]][val]


f, axs = plt.subplots(3, 4, sharey=True, figsize=(20, 10))
axs = [x for sublist in axs for x in sublist]  # flatten
f_orig, axs_orig = plt.subplots(3, 4, sharey=True, figsize=(20, 10))
axs_orig = [x for sublist in axs_orig for x in sublist]  # flatten

f_corr, axs_corr = plt.subplots(3, 4, sharey=True, figsize=(20, 10))
axs_corr = [x for sublist in axs_corr for x in sublist]  # flatten

f.suptitle("Probability (y) of chosen, if displayed")
f_orig.suptitle("Probability (y) of chosen, if displayed (original image was present)")
f_corr.suptitle("Probability (y) of smaller chosen given relative distance (x) between images")

params = sorted(parameter_range.keys(), key=lambda key: binom_test(analyzeDict[key]["smallerChosen"], n=analyzeDict[key]["smallerChosen"] + analyzeDict[key]["largerChosen"]))
for i, key in enumerate(params):
    print(f"{key}:\t{analyzeDict[key]['overall']} | {(analyzeDict[key]['overall'] / len(clean_df)) * 100}")
    print(
        "\tbinomial test overall w/o unsure:\tp: {:05.4f}".format(binom_test(analyzeDict[key]["smallerChosen"], n=analyzeDict[key]["smallerChosen"] + analyzeDict[key]["largerChosen"])),
        f"(x={analyzeDict[key]['smallerChosen']} | n={analyzeDict[key]['smallerChosen'] + analyzeDict[key]['largerChosen']})",
    )
    print(
        "\tbinomial test w/ orig. img. w/o unsure:\tp: {:05.4f}".format(binom_test(analyzeDict[key]["smallerChosenOrigPresent"], n=analyzeDict[key]["smallerChosenOrigPresent"] + analyzeDict[key]["largerChosenOrigPresent"])),
        f"(x={analyzeDict[key]['smallerChosenOrigPresent']} | n={analyzeDict[key]['smallerChosenOrigPresent'] + analyzeDict[key]['largerChosenOrigPresent']})",
    )
    if "leftNIMA" in sub_df.columns and "rightNIMA" in sub_df.columns:
        print("\tsurvey == NIMA: {:04.2f}%".format(analyzeDict[key]["sameNIMA"][True] / (analyzeDict[key]["sameNIMA"][True] + analyzeDict[key]["sameNIMA"][False]) * 100))
        pass

    print(f"\tsmaller edit:\t\t{'{:.1f}%'.format(analyzeDict[key]['smallerChosenOrigPresent'] / (analyzeDict[key]['smallerChosenOrigPresent'] + analyzeDict[key]['largerChosenOrigPresent']) * 100)}\t| {analyzeDict[key]['smallerChosenOrigPresent']}")
    print(f"\tlarger edit:\t\t{'{:.1f}%'.format(analyzeDict[key]['largerChosenOrigPresent'] / (analyzeDict[key]['smallerChosenOrigPresent'] + analyzeDict[key]['largerChosenOrigPresent']) * 100)}\t| {analyzeDict[key]['largerChosenOrigPresent']}")
    print(f"\tunsure and equal:\t{'{:.1f}%'.format(analyzeDict[key]['unsure_eq'] / analyzeDict[key]['overall'] * 100)}\t| {analyzeDict[key]['unsure_eq']}")
    print(f"\tunsure but not equal:\t{'{:.1f}%'.format(analyzeDict[key]['unsure_not_eq'] / analyzeDict[key]['overall'] * 100)}\t| {analyzeDict[key]['unsure_not_eq']}")
    print(f"\tnot unsure but equal:\t{'{:.1f}%'.format(analyzeDict[key]['not_unsure_eq'] / analyzeDict[key]['overall'] * 100)}\t| {analyzeDict[key]['not_unsure_eq']}")

    print("\tcorr. for pos. changes | one image original | larger relative changes == more clicks for original image?:")
    print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentPos"].items())))))
    print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentPos"].items())))))
    print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentPos"].items())))))

    if len(analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"]) != 0 and key != "vibrance":
        print("\tcorr. for neg. changes | one image original | larger relative changes == more clicks for original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"].items())))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"].items())))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"].items())))))

    print("\tcorr. for pos. changes | all | larger relative changes == more clicks for (more) original image?:")
    print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesPos"].items())))))
    print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesPos"].items())))))
    print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*analyzeDict[key]["smallerChosenRelChangesPos"].items())))))

    if len(analyzeDict[key]["smallerChosenRelChangesNeg"]) != 0 and key != "vibrance":
        print("\tcorr. for neg. changes | all | larger relative changes == more clicks for (more) original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesNeg"].items())))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*analyzeDict[key]["smallerChosenRelChangesNeg"].items())))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*analyzeDict[key]["smallerChosenRelChangesNeg"].items())))))

    axs_corr[i].set_title(key)
    axs_corr[i].set_ylim(bottom=0, top=1)
    tmp = list(zip(*sorted(analyzeDict[key]["smallerChosenRelChangesPos"].items(), key=lambda k: k[0])))
    axs_corr[i].plot(tmp[0], tmp[1], "-x", color="blue", label="click percentage per editing distance")
    tmp = list(zip(*sorted(analyzeDict[key]["smallerChosenRelChangesOrigPresentPos"].items(), key=lambda k: k[0])))
    axs_corr[i].plot(tmp[0], tmp[1], "-x", color="orange", label="click percentage per editing distance, original present")

    # axs_corr[i].grid(True, which="both") # FIXME?
    # axs_corr[i].minorticks_on()
    axs_corr[i].yaxis.set_ticks([0, 0.5, 1])
    axs_corr[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    axs_corr[i].yaxis.set_ticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], minor=True)
    axs_corr[i].yaxis.set_minor_formatter(ticker.FormatStrFormatter("%0.1f"))
    # axs_corr[i].xaxis.set_ticks([])
    # axs_corr[i].xaxis.set_ticks([])

    if len(analyzeDict[key]["smallerChosenRelChangesNeg"]) > 1:
        tmp = list(zip(*sorted(analyzeDict[key]["smallerChosenRelChangesNeg"].items(), key=lambda k: k[0])))
        tmp[0] = [val * -1 for val in tmp[0]]
        axs_corr[i].plot(tmp[0], tmp[1], "-x", color="blue")

        tmp = list(zip(*sorted(analyzeDict[key]["smallerChosenRelChangesOrigPresentNeg"].items(), key=lambda k: k[0])))
        tmp[0] = [val * -1 for val in tmp[0]]
        axs_corr[i].plot(tmp[0], tmp[1], "-x", color="orange")

    axs_corr[i].axhline(y=0.5, linestyle="-", color="grey", label="equally likely clicked")
    axs_corr[i].axvline(x=0, linestyle="--", color="orange", label="original image")

    axs[i].set_title(key)
    axs[i].set_xlim(left=min(parameter_range[key]["range"]), right=max(parameter_range[key]["range"]))
    axs[i].set_ylim(bottom=0, top=1)
    tmp = list(zip(*sorted(analyzeDict[key]["chosenChangesPos"].items(), key=lambda k: k[0])))
    axs[i].plot(tmp[0], tmp[1], "-x", color="blue", label="probability of chosen if displayed")
    sns.regplot(tmp[0], tmp[1], scatter=False, color="orange", label="linear regression", ax=axs[i])

    axs[i].grid(True, which="both")
    axs[i].minorticks_on()
    axs[i].yaxis.set_ticks([0, 0.5, 1])
    axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    axs[i].yaxis.set_ticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], minor=True)
    axs[i].yaxis.set_minor_formatter(ticker.FormatStrFormatter("%0.1f"))
    axs[i].xaxis.set_ticks([])
    axs[i].xaxis.set_ticks(parameter_range[key]["range"], minor=True)
    axs[i].xaxis.set_ticklabels(parameter_range[key]["range"], minor=True)

    tmp = list(zip(*sorted(analyzeDict[key]["chosenChangesNeg"].items(), key=lambda k: k[0])))
    if len(tmp) > 1:
        axs[i].plot(tmp[0], tmp[1], "-x", color="blue")
        sns.regplot(tmp[0], tmp[1], scatter=False, color="orange", ax=axs[i])

    axs[i].axhline(y=0.5, linestyle="-", color="grey", label="equally likely clicked")
    axs[i].axvline(x=parameter_range[key]["default"], linestyle="--", color="orange", label="original image")

    axs_orig[i].set_title(key)
    axs_orig[i].set_xlim(left=min(parameter_range[key]["range"]), right=max(parameter_range[key]["range"]))
    axs_orig[i].set_ylim(bottom=0, top=1)
    tmp = list(zip(*sorted(analyzeDict[key]["chosenChangesOrigPresentPos"].items(), key=lambda k: k[0])))
    axs_orig[i].plot(tmp[0], tmp[1], "-x", color="blue", label="probability of chosen if displayed")
    sns.regplot(tmp[0], tmp[1], scatter=False, color="orange", label="linear regression", ax=axs_orig[i])

    axs_orig[i].grid(True, which="both")
    axs_orig[i].minorticks_on()
    axs_orig[i].yaxis.set_ticks([0, 0.5, 1])
    axs_orig[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    axs_orig[i].yaxis.set_ticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], minor=True)
    axs_orig[i].yaxis.set_minor_formatter(ticker.FormatStrFormatter("%0.1f"))
    axs_orig[i].xaxis.set_ticks([])
    axs_orig[i].xaxis.set_ticks(parameter_range[key]["range"], minor=True)
    axs_orig[i].xaxis.set_ticklabels(parameter_range[key]["range"], minor=True)

    tmp = list(zip(*sorted(analyzeDict[key]["chosenChangesOrigPresentNeg"].items(), key=lambda k: k[0])))
    if len(tmp) > 1:
        axs_orig[i].plot(tmp[0], tmp[1], "-x", color="blue")
        sns.regplot(tmp[0], tmp[1], scatter=False, color="orange", ax=axs_orig[i])
    axs_orig[i].axhline(y=0.5, linestyle="-", color="grey", label="equally likely clicked")
    axs_orig[i].axvline(x=parameter_range[key]["default"], linestyle="--", color="orange", label="original image")

    f_small_vs, axs_small_vs = plt.subplots(math.ceil(math.sqrt(len(parameter_range[key]["range"]))), math.ceil(math.sqrt(len(parameter_range[key]["range"]))), sharey=True, figsize=(20, 10))
    axs_small_vs = [x for sublist in axs_small_vs for x in sublist]  # flatten
    f_small_vs.suptitle(f"{key}: probability (y) of chosen, if displayed against other (x)")

    for j, valSmall in enumerate(parameter_range[key]["range"]):
        x = []
        y = []
        if not valSmall in analyzeDict[key]["asSmallerChosenVS"]:
            x = parameter_range[key]["range"]
            y = [0] * len(parameter_range[key]["range"])
        else:
            for valLarge in parameter_range[key]["range"]:
                x.append(valLarge)
                y.append(analyzeDict[key]["asSmallerChosenVS"][valSmall].get(valLarge, 0))

        axs_small_vs[j].set_title(valSmall)
        axs_small_vs[j].set_xlim(left=min(parameter_range[key]["range"]), right=max(parameter_range[key]["range"]))
        axs_small_vs[j].set_ylim(bottom=0, top=1)
        axs_small_vs[j].plot(x, y, "-x", color="green", label="probability of chosen if displayed against")
        # axs_small_vs[j].axhline() # TODO
        axs_small_vs[j].axvline(x=valSmall, linestyle="--", color="orange", label="compared")

    for j, valLarge in enumerate(parameter_range[key]["range"]):
        x = []
        y = []
        if not valLarge in analyzeDict[key]["asLargerChosenVS"]:
            x = parameter_range[key]["range"]
            y = [0] * len(parameter_range[key]["range"])
        else:
            for valSmall in parameter_range[key]["range"]:
                x.append(valSmall)
                y.append(analyzeDict[key]["asLargerChosenVS"][valLarge].get(valSmall, 0))

        axs_small_vs[j].plot(x, y, "-x", color="red", label="probability of chosen if displayed against")

    f_small_vs.tight_layout()
    f_small_vs.savefig(plot_dir / "small_vs" / f"{key}.png")

    print()

f.tight_layout()
f.savefig(plot_dir / f"prob.png")

f_orig.tight_layout()
f_orig.savefig(plot_dir / f"prob_orig.png")

f_corr.tight_layout()
f_corr.savefig(plot_dir / f"corr.png")


print("---")
print()

plt.figure()
print("decision duration:")
durations = (sub_df.submitTime - sub_df.loadTime).astype("timedelta64[s]")

no_afk_durations = durations[durations < 60]
print(f"average time for decision: {'{:.1f}'.format(no_afk_durations.mean())} seconds")

plt.hist(durations.values, bins=range(0, int(no_afk_durations.max()) + 1), align="left")
# sns.distplot(durations, bins=range(0, int(no_afk_durations.max()) + 1), hist_kws={"align": "left"})
plt.ylim(bottom=0)
plt.xlim(left=-1, right=int(no_afk_durations.max()) + 2)
plt.tight_layout()
plt.savefig(plot_dir / "decision-duration.png")
plt.clf()

print("---")
print()

plt.figure()
print("useragent distribution:")
useragents = []
for _, row in sub_df.iterrows():
    useragents.append(httpagentparser.detect(row["useragent"]))

browser_count = collections.Counter([val["browser"]["name"] for val in useragents])
os_count = collections.Counter([val["os"]["name"] for val in useragents])
dist_count = collections.Counter([val["dist"]["name"] for val in useragents if "dist" in val])

print(browser_count)
print(os_count)
print(dist_count)
print("---")
print()

print("Top 5 longest sessions:")
usercount = sub_df[["userid", "hashval"]].rename(columns={"hashval": "count"}).groupby("userid").count()
print(usercount.nlargest(5, "count"))

plt.hist(usercount.values, bins=range(0, int(usercount.max()) + 1, 10), align="left")
# sns.distplot(usercount, bins=range(0, int(usercount.max()) + 1, 10), hist_kws={"align": "left"})
plt.ylim(bottom=0)
plt.xlim(left=-1, right=int(usercount.max()) + 1)
plt.tight_layout()
plt.savefig(plot_dir / "session-duration.png")
plt.clf()

print("---")
print()

print("3 most recent comparisons:")
print(sub_df.tail(3))

print("---")
print()
