import random
import argparse
from pathlib import Path

from skimage import exposure, io
from PIL import Image
import pillow_lut as lut


parser = argparse.ArgumentParser()
parser.add_argument("--imageFile", type=str, help="every line a file name", default="/scratch/stud/pfister/NIAA/pexels/train.txt")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images", default="/scratch/stud/pfister/NIAA/pexels/images")
parser.add_argument("--surveySize", type=int, help="how many comparisons to ask for", default="300")
parser.add_argument("--imageFolder", type=str, help="path to a folder of images", default="/scratch/stud/pfister/NIAA/pexels/images")
parser.add_argument("--out", type=str, help="dest for edited images", default="/home/stud/pfister/eclipse-workspace/NIAA/edit_survey/images")
args = parser.parse_args()

with open(args.imageFile, "r") as train:
    trainset = train.readlines()
