with open("/home/stud/pfister/eclipse-workspace/NIAA/model/margins.csv") as f:
    content = f.readlines()

content = [line.strip().split(",") for line in content]

distances = []

score = []

for line in content:
    distances.append(float(line[0]) - float(line[1]))
    score.append(float(line[0]))

for i in [0.51, 0.55, 0.60, 0.65]:
    print(i, "\t", sorted(distances)[round(i * len(distances))])

# 0.51     0.027368545532226562
# 0.55     0.10739612579345703
# 0.60     0.21038389205932617
# 0.65     0.3184628486633301
