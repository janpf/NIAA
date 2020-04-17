import math
import collections
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
from scipy.stats import pearsonr, spearmanr, wilcoxon, linregress
import httpagentparser
import redis

sys.path.insert(0, ".")
from edit_image import parameter_range

submission_log = "/scratch/stud/pfister/NIAA/pexels/logs/submissions.log"
#submission_log = "/home/stud/pfister/random.log"
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
    chosenDist[key]["posCorrelationBase"] = []  #  type: List[Tuple[float, int]]
    chosenDist[key]["negCorrelationBase"] = []  #  type: List[Tuple[float, int]]
    chosenDist[key]["posCorrelation"] = []  #  type: List[Tuple[float, int]]
    chosenDist[key]["negCorrelation"] = []  #  type: List[Tuple[float, int]]

for _, row in sub_df.iterrows():  # TODO feste variablen zuweisen innerhalb der schleife "smallerEdit", "largerEdit"
    if row["chosen"] == "error":
        continue

    parameter = row["parameter"]
    paramData = parameter_range[parameter]
    lChange = float(row["leftChanges"])  # TODO bei vergleichen immer "and not isclose"
    rChange = float(row["rightChanges"])
    chosen = row["chosen"]

    if lChange > paramData["default"]:
        if rChange < paramData["default"]:
            raise ("wait, that's illegal")
    if lChange < paramData["default"]:
        if rChange > paramData["default"]:
            raise ("wait, that's illegal")

    if math.isclose(lChange, rChange):
        if chosen == "unsure":
            chosenDict[parameter]["unsure_eq"] += 1
        else:
            chosenDict[parameter]["not_unsure_eq"] += 1
    else:
        if chosen == "unsure":
            chosenDict[parameter]["unsure_not_eq"] += 1

    if chosen == "unsure":
        pass
    elif chosen == "leftImage":
        if lChange < paramData["default"]:
            if lChange < rChange:
                chosenDict[parameter]["bigger"] += 1
            else:
                chosenDict[parameter]["smaller"] += 1
        else:
            if lChange < rChange:
                chosenDict[parameter]["smaller"] += 1
            else:
                chosenDict[parameter]["bigger"] += 1

    elif chosen == "rightImage":
        if lChange < paramData["default"]:
            if lChange < rChange:
                chosenDict[parameter]["smaller"] += 1
            else:
                chosenDict[parameter]["bigger"] += 1
        else:
            if lChange < rChange:
                chosenDict[parameter]["bigger"] += 1
            else:
                chosenDict[parameter]["smaller"] += 1
    else:
        raise ("wait, that's illegal")

    if chosen != "unsure" and not math.isclose(lChange, rChange):
        chosenDist[parameter]["displayed"][lChange] += 1
        chosenDist[parameter]["displayed"][rChange] += 1

        if chosen == "leftImage":
            chosenDist[parameter]["chosen"][lChange] += 1
        else:
            chosenDist[parameter]["chosen"][rChange] += 1

    if not math.isclose(lChange, rChange):
        if not (math.isclose(lChange, paramData["default"]) or math.isclose(rChange, paramData["default"])):
            continue
        if lChange > paramData["default"]:
            if lChange < rChange:
                if chosen == "leftImage":
                    chosenDist[parameter]["posCorrelation"].append((abs(rChange - lChange), 1))
                elif chosen == "rightImage":
                    chosenDist[parameter]["posCorrelation"].append((abs(rChange - lChange), 0))
            else:
                if chosen == "leftImage":
                    chosenDist[parameter]["posCorrelation"].append((abs(rChange - lChange), 0))
                elif chosen == "rightImage":
                    chosenDist[parameter]["posCorrelation"].append((abs(rChange - lChange), 1))
        elif lChange < paramData["default"]:
            if lChange < rChange:
                if chosen == "leftImage":
                    chosenDist[parameter]["negCorrelation"].append((abs(rChange - lChange), 0))
                elif chosen == "rightImage":
                    chosenDist[parameter]["negCorrelation"].append((abs(rChange - lChange), 1))
            else:
                if chosen == "leftImage":
                    chosenDist[parameter]["negCorrelation"].append((abs(rChange - lChange), 1))
                elif chosen == "rightImage":
                    chosenDist[parameter]["negCorrelation"].append((abs(rChange - lChange), 0))

params = sorted(parameter_range.keys(), key=lambda k: chosenDict[k]["bigger"] / chosenDict[k]["smaller"])
for key in params:
    print(f"{key}:\t{'{:.1f}%'.format(sum(chosenDict[key].values()) / sum([sum(val.values()) for val in chosenDict.values()])*100)}\t| {sum(chosenDict[key].values())}")
    print(f"\tsmaller edit:\t\t{'{:.1f}%'.format(chosenDict[key]['smaller'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['smaller']}")
    print(f"\tlarger edit:\t\t{'{:.1f}%'.format(chosenDict[key]['bigger'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['bigger']}")
    print(f"\tunsure and equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_eq']}")
    print(f"\tunsure but not equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_not_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_not_eq']}")
    print(f"\tnot unsure but equal:\t{'{:.1f}%'.format(chosenDict[key]['not_unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['not_unsure_eq']}")

    try:
        print("\tcorr. for pos. changes | one image original | bigger changes == more clicks for original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["posCorrelation"])))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["posCorrelation"])))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["posCorrelation"])))))
    except:
        print(chosenDist[key]["posCorrelation"])
    if len(chosenDist[key]["negCorrelation"]) != 0:
        print("\tcorr. for neg. changes | one image original | bigger changes == more clicks for original image?:")
        print("\t\tpearson:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*pearsonr(*list(zip(*chosenDist[key]["negCorrelation"])))))
        print("\t\tspearman:\tcorr. coeff: {:05.3f} p: {:05.4f}".format(*spearmanr(*list(zip(*chosenDist[key]["negCorrelation"])))))
        print("\t\tlinregr:\tslope: {:05.3f} intercept: {:05.3f} corr. coeff: {:05.3f} p: {:05.4f} stderr: {:05.3f}".format(*linregress(*list(zip(*chosenDist[key]["negCorrelation"])))))

    x = []
    y = []
    for k, v in sorted(chosenDist[key]["chosen"].items(), key=lambda k: k[0]):
        x.append(k)
        y.append((chosenDist[key]["chosen"][k] / chosenDist[key]["displayed"][k]) * 100)
    plt.plot(x, y, "-x")
    plt.axvline(x=parameter_range[key]["default"], linestyle="--", color="orange")
    plt.ylim(bottom=0)
    plt.savefig(plot_dir / f"{key}_dist.png")
    plt.clf()
    # print(sorted(list(chosenDist[key].items()), key=lambda k: k[0]))
    print()
print("---")
print()


print("decision duration:")
durations = (sub_df.submitTime - sub_df.loadTime).astype("timedelta64[s]")

no_afk_durations = durations[durations < 60]
print(f"average time for decision: {'{:.1f}'.format(no_afk_durations.mean())} seconds")

plt.hist(durations.values, bins=range(0, int(no_afk_durations.max()) + 1), align="left")
plt.ylim(bottom=0)
plt.savefig(plot_dir / "decision-duration.png")
plt.clf()

print("---")
print()


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
plt.ylim(bottom=0)
plt.savefig(plot_dir / "session-duration.png")
plt.clf()

print("---")
print()


print("3 most recent comparisons:")
print(sub_df.tail(3))

print("---")
print()
