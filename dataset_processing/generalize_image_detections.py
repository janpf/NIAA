from collections import Counter
from nltk.corpus import wordnet as wn

with open("/home/janpf/projects/NIAA/analysis/not_uploaded/detected_in_images.csv") as f:
    content = f.readlines()

content = [val.strip() for val in content]
content = [val.split(";")[1:] for val in content]
content = [val for val in content if len(val) != 0]
content = [[obj.split(":") for obj in val] for val in content]
content = [[[obj[0], float(obj[1]), eval(obj[2])] for obj in val] for val in content]

print(f"number of images with objects: {len(content)}")
# print(f"average number of objects per image: {}")

topchoices = Counter([obj[0][0] for obj in content])
print(topchoices.most_common(5))

# fmt: off
imagenet_topclasses = [
    "plant", "flora", "plant_life",
    "geological_formation", "formation",
    "natural_object",
    "sport", "athletics",
    "artifact", "artefact",
    "fungus",
    "person", "individual", "someone", "somebody", "mortal", "soul",
    "animal", "animate_being", "beast", "brute", "creature", "fauna"
    ]
# fmt: on

imagenet_topclasses_counter = dict()

for val in imagenet_topclasses:
    imagenet_topclasses_counter[val] = 0

hierarchies = []

for word in topchoices.most_common():
    word_hierarchy = []
    word = word[0]
    word = f"{word}.n.01"
    word_hierarchy.append(word)
    while True:
        try:
            word = wn.synset(word).hypernyms()[0].name()
            word_hierarchy.append(word)
        except:
            break
    word_hierarchy = [val[: val.index(".")] for val in word_hierarchy]
    hierarchies.append(word_hierarchy)

# print(hierarchies)

for hier in hierarchies:
    current_classes = []
    for val in imagenet_topclasses:
        if val in hier:
            current_classes.append(val)
    if len(current_classes) > 1:
        print(f"{hier[0]} is in classes {current_classes}")
    elif len(current_classes) == 0:
        print(f"{hier[0]} is in no class: {hier}")
    if len(current_classes) > 0:
        # choosing top most class
        imagenet_topclasses_counter[current_classes[-1]] += topchoices[hier[0]]

for clazz in list(imagenet_topclasses_counter.keys()):
    if imagenet_topclasses_counter[clazz] == 0:
        del imagenet_topclasses_counter[clazz]

print(imagenet_topclasses_counter)
# {'plant': 489, 'natural_object': 1208, 'artifact': 25656, 'person': 44308, 'animal': 8453}
