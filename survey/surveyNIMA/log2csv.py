import json
from pathlib import Path

import pandas as pd

submission_log = "/scratch/stud/pfister/NIAA/pexels/logs/submissions.log"
# submission_log = "/home/stud/pfister/random.log"
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
sub_df.to_csv(out_file)
