import math
import collections
from dateutil import parser
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import sys
import httpagentparser
import redis

sys.path.insert(0, ".")
from edit_image import parameter_range

sqlite_db = "/scratch/stud/pfister/NIAA/pexels/logs/submissions.db"  # "/data/logs/submissions.db"
plot_dir = Path.home() / "eclipse-workspace" / "NIAA" / "analysis" / "survey"  # type: Path

plot_dir.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(sqlite_db)
conn.row_factory = sqlite3.Row
c = conn.cursor()
sdata = c.execute("""SELECT * FROM submissions ORDER BY id""").fetchall()
usercount = c.execute("""SELECT userid, COUNT(id), useragent FROM submissions GROUP BY userid ORDER BY COUNT(id) DESC""").fetchall()
choicecount = c.execute("""SELECT chosen, COUNT(id) as count FROM submissions GROUP BY chosen ORDER BY chosen""").fetchall()
conn.commit()
conn.close()

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


print(f"{len(sdata)} images compared in {len(usercount)} sessions")
print("overall distribution:")
for row in choicecount:
    print(f"\t{row['chosen']}: {row['count']}")
print("---")
print()


for row in sdata:
    if row["chosen"] == "error":
        break
        print(tuple(row))

chosenDict = dict()
for key in parameter_range.keys():
    chosenDict[key] = collections.defaultdict(lambda: 0)

for row in sdata:
    if row["chosen"] == "error":
        continue

    paramData = parameter_range[row["parameter"]]
    lChange = float(row["leftChanges"])
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
            chosenDict[row["parameter"]]["unsure_eq"] += 1
        else:
            chosenDict[row["parameter"]]["not_unsure_eq"] += 1
    else:
        if chosen == "unsure":
            chosenDict[row["parameter"]]["unsure_not_eq"] += 1

    if chosen == "unsure":
        pass
    elif chosen == "leftImage":
        if lChange < paramData["default"]:
            if lChange < rChange:
                chosenDict[row["parameter"]]["bigger"] += 1
            else:
                chosenDict[row["parameter"]]["smaller"] += 1
        else:
            if lChange < rChange:
                chosenDict[row["parameter"]]["smaller"] += 1
            else:
                chosenDict[row["parameter"]]["bigger"] += 1

    elif chosen == "rightImage":
        if lChange < paramData["default"]:
            if lChange < rChange:
                chosenDict[row["parameter"]]["smaller"] += 1
            else:
                chosenDict[row["parameter"]]["bigger"] += 1
        else:
            if lChange < rChange:
                chosenDict[row["parameter"]]["bigger"] += 1
            else:
                chosenDict[row["parameter"]]["smaller"] += 1
    else:
        raise ("wait, that's illegal")

params = sorted(parameter_range.keys(), key=lambda k: chosenDict[k]["bigger"] / chosenDict[k]["smaller"])
for key in params:
    print(f"{key}:\t{'{:.1f}%'.format(sum(chosenDict[key].values()) / sum([sum(val.values()) for val in chosenDict.values()])*100)}\t| {sum(chosenDict[key].values())}")
    print(f"\tsmaller edit:\t\t{'{:.1f}%'.format(chosenDict[key]['smaller'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['smaller']}")
    print(f"\tbigger edit:\t\t{'{:.1f}%'.format(chosenDict[key]['bigger'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['bigger']}")
    print(f"\tunsure and equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_eq']}")
    print(f"\tunsure but not equal:\t{'{:.1f}%'.format(chosenDict[key]['unsure_not_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['unsure_not_eq']}")
    print(f"\tnot unsure but equal:\t{'{:.1f}%'.format(chosenDict[key]['not_unsure_eq'] / sum(chosenDict[key].values()) * 100)}\t| {chosenDict[key]['not_unsure_eq']}")
print("---")
print()


print("decision duration")
durations = []
for row in sdata:
    loadTime = parser.parse(row["loadTime"])
    submTime = parser.parse(row["submitTime"])
    duration = (submTime - loadTime).total_seconds()
    durations.append(duration)

no_afk_durations = [val for val in durations if val < 60]
print(f"average time for decision: {'{:.1f}'.format(sum(no_afk_durations)/len(no_afk_durations))} seconds")

plt.hist(durations, bins=range(0, int(max([val for val in durations if val < 60])) + 1))
plt.ylim(bottom=0)
plt.savefig(plot_dir / "decision-duration.png")
plt.clf()

print("---")
print()


print("useragent distribution:")
useragents = []
for row in usercount:
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
for row in usercount[:5]:
    print(f"\t{row[0]}: {row[1]}")

plt.hist([val[1] for val in usercount], bins=range(0, max([val[1] for val in usercount]) + 1, 10))
plt.ylim(bottom=0)
plt.savefig(plot_dir / "session-duration.png")
plt.clf()

print("---")
print()


print("3 most recent comparisons:")
for row in sdata[-3:]:
    print(tuple(row))

print("---")
print()
