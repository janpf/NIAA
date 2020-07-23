import numpy as np

with open("margins.csv") as f:
    content = f.readlines()

content = [line.strip().split(",") for line in content]

distances = []

score = []

for line in content:
    distances.append(float(line[0]) - float(line[1]))
    score.append(float(line[0]))

for i in [0.51, 0.55, 0.6, 0.65]:
    print(i, "\t", sorted(distances)[round(i * len(distances))])
