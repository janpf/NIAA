with open("margins.csv") as f:
    content = f.readlines()

content = [line.strip().split(",") for line in content]

distances = []

for line in content:
    distances.append(float(line[0]) - float(line[1]))

distances = [abs(val) for val in distances]

print("avg dist:", sum(distances) / len(distances))
