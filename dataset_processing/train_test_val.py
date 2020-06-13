import random
from pathlib import Path
import csv
from collections import Counter

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
survey_counter = Counter(survey_imgs)
survey_imgs_count = sorted(list(dict(survey_counter).items()), key=lambda k: k[1], reverse=True)
survey_once = [val[0] for val in survey_imgs_count if val[1] == 1]
survey_more_than_once = [val[0] for val in survey_imgs_count if val[1] != 1]
random.shuffle(survey_more_than_once)

print(f"images in survey:{len(survey_imgs)}")
print(f"images in survey once:{len(survey_once)}")
print(f"images in survey more than once:{len(survey_more_than_once)}")

with open("/home/stud/pfister/eclipse-workspace/pexels-scraper/urls.txt") as f:
    all_imgs = f.readlines()

all_imgs: set = {val.replace("https://", "").strip() for val in all_imgs}
all_imgs = {Path(val).name for val in all_imgs}

print(f"images existing: {len(all_imgs)}")
all_imgs = set(all_imgs)
print(f"images existing (unique): {len(all_imgs)}")
all_imgs = {val for val in all_imgs if val not in ignored}
print(f"images existing (after ignored): {len(all_imgs)}")
all_imgs_rest = {val for val in all_imgs if val not in survey_imgs}
print(f"images existing (without survey): {len(all_imgs_rest)}")

train_survey_percentage = 0.7
train_count = 100000
val_count = test_count = 15000

train_set = survey_more_than_once[len(survey_more_than_once) // 2 :]
test_set = survey_more_than_once[: len(survey_more_than_once) // 2]

tmp = [val for val in train_set]
train_set = []

for val in tmp:
    train_set.extend([val]*survey_counter[val])

tmp = [val for val in test_set]
test_set = []
for val in tmp:
    test_set.extend([val]*survey_counter[val])
del tmp

for img in survey_once:
    if len(train_set) < train_survey_percentage * len(survey_imgs):
        train_set.append(img)
    else:
        test_set.append(img)

train_set = set(train_set)
test_set = set(test_set)

rest = list(all_imgs.difference(survey_imgs))
random.shuffle(rest)
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

