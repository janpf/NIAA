import json
from pathlib import Path
from collections import Counter
import pandas as pd

submission_log = Path.home() / "eclipse-workspace" / "NIAA" / "survey" / "submissions.log"  # type: Path
out_file = Path.home() / "eclipse-workspace" / "NIAA" / "survey" / "survey.csv"  # type: Path

with open(submission_log, mode="r") as subs_file:
    subs = subs_file.readlines()

subs = (row.strip() for row in subs)
subs = (row.split("submit:")[1] for row in subs)
subs = (row.strip() for row in subs)
subs = (row.replace("'id':", "'userid':") for row in subs)
subs = (row.replace("'", '"') for row in subs)
subs = [json.loads(row) for row in subs]

sub_df = pd.read_json(json.dumps(subs), orient="records", convert_dates=["loadTime", "submitTime"])  # type: pd.DataFrame
durations = (sub_df.submitTime - sub_df.loadTime).astype("timedelta64[s]")
durations = [int(val) for val in durations]
sub_df.insert(0, "RTT(s)", durations)
sub_df.drop(columns=["loadTime", "submitTime"], inplace=True)

duplicates = Counter(list(sub_df.hashval))

duplicates = {k: v for k, v in duplicates.items() if v > 1}
if len(duplicates) > 0:
    print("duplicates:")
for k, v in duplicates.items():
    rows = sub_df[(sub_df.hashval == k)]
    print()
    print(rows["chosen"])
    if len(set(rows["chosen"])) == 1 and len(set(rows["userid"])) == 1:
        print("kept")
    else:
        print("dropped")
        sub_df.drop(rows.index, inplace=True)
sub_df.to_csv(out_file, index=False)
