import random
from pathlib import Path
import csv
from collections import Counter

with open("/home/stud/pfister/eclipse-workspace/NIAA/dataset_processing/ignored_images.txt") as f:
    ignored = f.readlines()

ignored = {val.replace("\n", "") for val in ignored}
print(f"images ignored: {len(ignored)}")

survey_imgs = []
with open("/home/stud/pfister/eclipse-workspace/NIAA/survey/survey.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        survey_imgs.append(row[1])
survey_imgs = [Path(val).name for val in survey_imgs]
survey_imgs_count = sorted(list(dict(Counter(survey_imgs)).items()), key=lambda k: k[1], reverse=True)

print(f"images in survey:{len(survey_imgs)}")

with open("/home/stud/pfister/eclipse-workspace/pexels-scraper/urls.txt") as f:
    all_imgs = f.readlines()

all_imgs = [val.replace("https://", "").replace("\n", "") for val in all_imgs]
all_imgs = [Path(val).name for val in all_imgs]
print(f"images existing: {len(all_imgs)}")
all_imgs = set(all_imgs)
print(f"images existing (unique): {len(all_imgs)}")
all_imgs = [val for val in all_imgs if val not in ignored]
print(f"images existing (afer ignored): {len(all_imgs)}")
