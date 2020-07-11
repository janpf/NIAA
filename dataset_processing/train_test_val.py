import csv
import random
from collections import Counter
from pathlib import Path

with open("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/ignored_images.txt") as f:
    ignored = f.readlines()

ignored = {val.strip() for val in ignored}
print(f"images ignored: {len(ignored)}")

survey_imgs: list = []
with open("/home/stud/pfister/eclipse-workspace/NIAA/survey/survey.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        survey_imgs.append(row[1])
survey_imgs = [Path(val).name for val in survey_imgs]
print(f"images in survey:{len(survey_imgs)}")
print(f"ignored images in survey:{len(set(survey_imgs).intersection(ignored))}")
survey_imgs = [val for val in survey_imgs if val not in ignored]
print(f"images in survey:{len(survey_imgs)}")

all_imgs = [f.name for f in Path("/scratch/stud/pfister/NIAA/pexels/images").iterdir()]

all_imgs: set = {val.replace("https://", "").strip() for val in all_imgs}
all_imgs = {Path(val).name for val in all_imgs}

print(f"images existing: {len(all_imgs)}")
all_imgs = set(all_imgs)
print(f"images existing (unique): {len(all_imgs)}")
all_imgs = {val for val in all_imgs if val not in ignored}
print(f"images existing (after ignored): {len(all_imgs)}")
all_imgs_rest = {val for val in all_imgs if val not in survey_imgs}
print(f"images existing (without survey): {len(all_imgs_rest)}")

train_count = 100000
val_count = test_count = 15000

test_set = set(survey_imgs)

rest = list(all_imgs.difference(survey_imgs))
random.shuffle(rest)
train_set = set()
val_set = set()


for img in rest:
    if len(train_set) < train_count:
        train_set.add(img)
    elif len(val_set) < val_count:
        val_set.add(img)
    elif len(test_set) < test_count:
        test_set.add(img)
    else:
        print("all distributed, done")
        break

print("checking")
print(f"trainset:\t{len(train_set)}")
print(f"valset:\t\t{len(val_set)}")
print(f"testset:\t{len(test_set)}")

print(f"survey train:\t{len([val for val in survey_imgs if val in train_set])}")
print(f"survey val:\t{len([val for val in survey_imgs if val in val_set])}")
print(f"survey test:\t{len([val for val in survey_imgs if val in test_set])}")

print(f"train -> test intersection: {len(train_set.intersection(test_set))}")
print(f"train -> val intersection: {len(train_set.intersection(val_set))}")
print(f"test -> val intersection: {len(test_set.intersection(val_set))}")

with open("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/train_set.txt", "w") as f:
    for val in sorted(list(train_set)):
        f.write(val + "\n")


with open("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/val_set.txt", "w") as f:
    for val in sorted(list(val_set)):
        f.write(val + "\n")


with open("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/test_set.txt", "w") as f:
    for val in sorted(list(test_set)):
        f.write(val + "\n")
