import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np
from mpl_toolkits import mplot3d

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, help="path to a csv")
parser.add_argument("--out", type=str, help="dest for edited images")
args = parser.parse_args()

data = []
for row in csv.reader(open(args.csv)):
    data.append([float(val) for val in row if not row[0] == "b"])
data = data[1:]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf([val[0] for val in data], [val[1] for val in data], [val[2] for val in data])  # , cmap=plt.cm.CMRmap
ax.set_xlabel("brightness")
ax.set_ylabel("contrast")
ax.set_zlabel("score")

ax.view_init(30, 260)
plt.savefig("b_c.png")

ax.view_init(30, 190)
plt.savefig("c_b.png")
