import collections
import json
import math
import sys
from pathlib import Path

import httpagentparser
import matplotlib.pyplot as plt
import pandas as pd
import redis
import seaborn as sns
from scipy.stats import binom_test, linregress, pearsonr, spearmanr, wilcoxon

sys.path.insert(0, ".")
from edit_image import parameter_range

sns.set(style="whitegrid")

submission_log = "/scratch/stud/pfister/NIAA/pexels/logs/submissions.log"
# submission_log = "/home/stud/pfister/random.log"
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


with open(submission_log, mode="r") as subs_file:
    subs = subs_file.readlines()

subs = (row.strip() for row in subs)
subs = (row.split("submit:")[1] for row in subs)
subs = (row.strip() for row in subs)
subs = (row.replace("'id':", "'userid':") for row in subs)
subs = (row.replace("'", '"') for row in subs)
subs = [json.loads(row) for row in subs]

sub_df = pd.read_json(json.dumps(subs), orient="records", convert_dates=["loadTime", "submitTime"])  # type: pd.DataFrame
# data reading done

print()
print(f"{sub_df.hashval.count()} images compared in {sub_df.groupby('userid').userid.count().count()} sessions")
print(sub_df.groupby("chosen").chosen.count().to_string())
print("---")
print()

chosenDist = dict()
chosenDict = dict()
for key in parameter_range.keys():
    chosenDict[key] = collections.defaultdict(lambda: 0)
    chosenDist[key] = dict()
    chosenDist[key]["chosen"] = collections.defaultdict(lambda: 0)
    chosenDist[key]["displayed"] = collections.defaultdict(lambda: 0)
    chosenDist[key]["posCorrelationBase"] = collections.defaultdict(lambda: 0)
    chosenDist[key]["negCorrelationBase"] = collections.defaultdict(lambda: 0)
    chosenDist[key]["posCorrelation"] = collections.defaultdict(lambda: 0)
    chosenDist[key]["negCorrelation"] = collections.defaultdict(lambda: 0)

for _, row in sub_df.iterrows():
    if row["chosen"] == "error":
        continue

    parameter = row["parameter"]
    paramData = parameter_range[parameter]
    lChange = float(row["leftChanges"])
    rChange = float(row["rightChanges"])

    if math.isclose(lChange, 0):
        lChange = 0
    if math.isclose(rChange, 0):
        rChange = 0

    if math.isclose(lChange, paramData["default"]):
        lChange = paramData["default"]
    if math.isclose(rChange, paramData["default"]):
        rChange = paramData["default"]

    lRelDistDefault = abs(paramData["default"] - lChange)
    rRelDistDefault = abs(paramData["default"] - rChange)

    chosen = row["chosen"]
    bothSame = math.isclose(lChange, rChange)

    if not math.isclose(lChange, rChange):
        if lChange < paramData["default"] and rChange > paramData["default"]:
            raise ("hä")
        if lChange > paramData["default"] and rChange < paramData["default"]:
            raise ("hä")

    if bothSame and math.isclose(lChange, paramData["default"]):
        changeSign = "|"  # both original

    if not bothSame:
        if lChange > paramData["default"] or rChange > paramData["default"]:
            changeSign = "+"
        else:
            changeSign = "-"

    if bothSame:
        smallChange = largeChange = lChange

    elif not bothSame:
        if lRelDistDefault < rRelDistDefault:
            smallChange = lChange
            largeChange = rChange
        elif lRelDistDefault > rRelDistDefault:
            smallChange = rChange
            largeChange = lChange
        else:
            raise ("hä")

    if math.isclose(smallChange, paramData["default"]):
        smallChangeIsOriginal = True
    else:
        smallChangeIsOriginal = False

    if math.isclose(largeChange, paramData["default"]):
        largeChangeIsOriginal = True
    else:
        largeChangeIsOriginal = False

    if not smallChangeIsOriginal and largeChangeIsOriginal:
        raise ("hä")

    if not bothSame:
        if math.isclose(lChange, smallChange):
            if chosen == "leftImage":
                smallerChosen = True
                largerChosen = False
            elif chosen == "rightImage":
                smallerChosen = False
                largerChosen = True
            else:
                smallerChosen = False
                largerChosen = False
        elif math.isclose(rChange, smallChange):
            if chosen == "leftImage":
                smallerChosen = False
                largerChosen = True
            elif chosen == "rightImage":
                smallerChosen = True
                largerChosen = False
            else:
                smallerChosen = False
                largerChosen = False
        else:
            raise ("hä")

    # actually analysing

    if bothSame:
        if chosen == "unsure":
            chosenDict[parameter]["unsure_eq"] += 1
        else:
            chosenDict[parameter]["not_unsure_eq"] += 1
    else:
        if chosen == "unsure":
            chosenDict[parameter]["unsure_not_eq"] += 1

    if not bothSame and chosen != "unsure":
        if smallerChosen:
            chosenDict[parameter]["smaller"] += 1
            if smallChangeIsOriginal:
                chosenDict[parameter]["origsmaller"] += 1
        elif largerChosen:
            chosenDict[parameter]["larger"] += 1
            if smallChangeIsOriginal:
                chosenDict[parameter]["origlarger"] += 1
        else:
            raise ("hä")

        chosenDist[parameter]["displayed"][lChange] += 1
        chosenDist[parameter]["displayed"][rChange] += 1

        if chosen == "leftImage":
            chosenDist[parameter]["chosen"][lChange] += 1
        else:
            chosenDist[parameter]["chosen"][rChange] += 1

    if not bothSame:
        if smallChangeIsOriginal:
            if smallerChosen:
                if changeSign == "+":
                    chosenDist[parameter]["posCorrelationBase"][abs(largeChange)] += 1
                elif changeSign == "-":
                    chosenDist[parameter]["negCorrelationBase"][abs(largeChange)] += 1
        else:
            if smallerChosen:
                if changeSign == "+":
                    chosenDist[parameter]["posCorrelation"][abs(largeChange - smallChange)] += 1
                elif changeSign == "-":
                    chosenDist[parameter]["negCorrelation"][abs(largeChange - smallChange)] += 1

f, axs = plt.subplots(3, 4, sharey=True, figsize=(20, 10))
axs = [x for sublist in axs for x in sublist]  # flatten
params = sorted(parameter_range.keys(), key=lambda k: binom_test(chosenDict[k]["smaller"], n=chosenDict[k]["smaller"] + chosenDict[k]["larger"]))
for i, key in enumerate(params):
    print(f"{key}:\t{'{:.1f}%'.format(sum(chosenDict[key].values()) / sum([sum(val.values()) for val in chosenDict.values()])*100)}\t| {sum(chosenDict[key].values())}")
    print("\tbinomial test overall w/o unsure:\tp: {:05.4f}".format(binom_test(chosenDict[key]["smaller"], n=chosenDict[key]["smaller"] + chosenDict[key]["larger"])), f"(x={chosenDict[key]['smaller']} | n={chosenDict[key]['smaller'] + chosenDict[key]['larger']})")
    print("\tbinomial test w/ orig. img. w/o unsure:\tp: {:05.4f}".format(binom_test(chosenDict[key]["origsmaller"], n=chosenDict[key]["origsmaller"] + chosenDict[key]["origlarger"])), f"(x={chosenDict[key]['origsmaller']} | n={chosenDict[key]['origsmaller'] + chosenDict[key]['origlarger']})")

    print(f"\tsmaller edit:\t\t{'{:.1f}%'.format(chosenDict[key]['smaller'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['smaller']}")
    print(f"\tlarger edit:\t\t{'{:.1f}%'.format(chosenDict[key]['larger'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['larger']}")
    print(f"\tunsure and equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_eq']}")
    print(f"\tunsure but not equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_not_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_not_eq']}")
    print(f"\tnot unsure but equal:\t{'{:.1f}%'.format(chosenDict[key]['not_unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['not_unsure_eq']}")

    print("\tcorr. for pos. changes | one image original | larger changes == more clicks for original image?:")
    print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["posCorrelationBase"].items())))))
    print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["posCorrelationBase"].items())))))
    print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["posCorrelationBase"].items())))))

    if len(chosenDist[key]["negCorrelationBase"]) != 0 and key != "vibrance":
        print("\tcorr. for neg. changes | one image original | larger changes == more clicks for original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["negCorrelationBase"].items())))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["negCorrelationBase"].items())))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["negCorrelationBase"].items())))))

    print("\tcorr. for pos. changes | all | larger changes == more clicks for original image?:")
    print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["posCorrelation"].items())))))
    print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["posCorrelation"].items())))))
    print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["posCorrelation"].items())))))

    if len(chosenDist[key]["negCorrelation"]) != 0 and key != "vibrance":
        print("\tcorr. for neg. changes | all | larger changes == more clicks for original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["negCorrelation"].items())))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["negCorrelation"].items())))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["negCorrelation"].items())))))

    x = []
    y = []
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []

    axs[i].set_title(key)

    for k, v in sorted(chosenDist[key]["chosen"].items(), key=lambda k: k[0]):
        x.append(k)
        y.append((chosenDist[key]["chosen"][k] / chosenDist[key]["displayed"][k]) * 100)

        if k >= parameter_range[key]["default"] or math.isclose(k, parameter_range[key]["default"]):
            x_pos.append(k)
            y_pos.append((chosenDist[key]["chosen"][k] / chosenDist[key]["displayed"][k]) * 100)
        if k <= parameter_range[key]["default"] or math.isclose(k, parameter_range[key]["default"]):
            x_neg.append(k)
            y_neg.append((chosenDist[key]["chosen"][k] / chosenDist[key]["displayed"][k]) * 100)

    axs[i].plot(x, y, "-x", label="probability of chosen if displayed")
    axs[i].axvline(x=parameter_range[key]["default"], linestyle="--", color="orange", label="original image")

    sns.regplot(x_pos, y_pos, scatter=False, color="orange", label="linear regression", ax=axs[i])
    if len(x_neg) > 1:
        sns.regplot(x_neg, y_neg, scatter=False, color="orange", ax=axs[i])

    axs[i].set_ylim(bottom=0, top=100)
    print()

plt.tight_layout()
plt.savefig(plot_dir / f"dist.png")
plt.clf()

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
