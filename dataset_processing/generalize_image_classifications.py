from collections import Counter

with open("/home/stud/pfister/eclipse-workspace/NIAA/analysis/dataset_classes.csv") as f:
    content = f.readlines()

content = [val.strip() for val in content]
content = [val.split(",") for val in content]

topchoices = Counter([val[1].split(":")[0] for val in content])

for choice in topchoices.most_common():
    print(f"{choice[1]}\t{choice[0]}")

exit()
# yeah no. just gonna do it manually
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from nltk.corpus import wordnet as wn

G = nx.DiGraph()
print(topchoices.most_common(10))
for word in topchoices.most_common(10):
    word = word[0]
    word = f"{word}.n.01"  # most likely noun i guess? # TODO check
    try:
        hypernym = wn.synset(word).hypernyms()[0].name()
    except:
        hypernym = "none"
    print(word, hypernym)
    G.add_edge(word, hypernym)
pos = graphviz_layout(G, prog="dot")
nx.draw(G, pos, with_labels=True)
plt.savefig("graph.png")
