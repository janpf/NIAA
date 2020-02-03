import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, help="path to a csv")
parser.add_argument("--out", type=str, help="folder to save image to")
args = parser.parse_args()

data = []
for row in csv.reader(open(args.csv)):
    data.append([float(val) for val in row if not row[0] == "change"])
data = data[1:]

plt.plot([val[0] for val in data], [val[1] for val in data])
plt.xlabel(Path(args.csv).stem)
plt.ylabel("score")

plt.savefig(f"{args.out}/{Path(args.csv).stem}.png")
