with open("/home/stud/pfister/eclipse-workspace/NIAA/model/margins.csv") as f:
    content = f.readlines()

content = [line.strip().split(",") for line in content]

distances = []

score = []

for line in content:
    distances.append(float(line[0]) - float(line[1]))
    score.append(float(line[0]))

print(0, "\t", sorted(distances)[0])

for i in [0.51, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
    print(i, "\t", sorted(distances)[round(i * len(distances))])

print(1, "\t", sorted(distances)[-1])


# 0        -7.726612091064453
# 0.51     0.027368545532226562
# 0.55     0.10739612579345703
# 0.6      0.21038389205932617
# 0.65     0.3184628486633301
# 0.7      0.43505430221557617
# 0.8      0.7166886329650879
# 0.9      1.1495742797851562
# 1        6.341530799865723
