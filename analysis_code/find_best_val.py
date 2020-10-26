import csv
from pathlib import Path
from collections import defaultdict

best_results = defaultdict(lambda: float("inf"))
best_results_path = dict()

with open("analysis/IA/vals.csv") as f:
    for row in csv.DictReader(f):
        row["loss"] = float(row["loss"])
        if row["loss"] < best_results[str(Path(row["path"]).parent)]:
            best_results[str(Path(row["path"]).parent)] = row["loss"]
            best_results_path[str(Path(row["path"]).parent)] = row["path"]

for k, v in best_results.items():
    print(f"{best_results_path[k]}\t {v}")
