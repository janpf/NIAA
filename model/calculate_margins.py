import math
import numpy as np
import random
import scipy.stats as st
import seaborn as sns

with open("margins.csv") as f:
    content = f.readlines()

content = [line.strip().split(",") for line in content]

distances = []

score = []

for line in content:
    distances.append(float(line[0]) - float(line[1]))
    score.append(float(line[0]))

print(sorted(distances)[round(0.51 * len(distances))])
exit()

print(len(distances))
print(distances[:5])

print("is normal:", st.normaltest(distances))
print("score is normal:", st.normaltest(score))

distances = [abs(val) for val in distances]
print(np.mean(distances))

exit()

print("median dist:", np.median(distances))
print("avg dist:", np.mean(distances))
print("std dev dist:", np.std(distances))

half_normal_std = [val * val for val in distances]  # https://en.wikipedia.org/wiki/Half-normal_distribution#Parameter_estimation
half_normal_std = sum(half_normal_std)
half_normal_std = (1 / len(distances)) * half_normal_std
half_normal_std = math.sqrt(half_normal_std)

bias = -(half_normal_std / (4 * len(distances)))
half_normal_std = half_normal_std - bias
print("half norm std dev dist:", half_normal_std)

nd = np.random.normal(0, half_normal_std, len(distances))
nd = [abs(val) for val in nd]

print(len(distances))
print(len(nd))

sns.distplot(distances).get_figure().savefig("margins.png")
# sns.distplot(nd).get_figure().savefig("nd.png")

# exit()
std = np.std(distances)
# std = half_normal_std
scale = 1 / np.sqrt(2) * (2 * std)
# scale = std
fnd = st.foldnorm(loc=0, c=0, scale=scale)
sns.distplot(fnd.rvs(size=len(distances))).get_figure().savefig("nd.png")
