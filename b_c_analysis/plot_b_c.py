import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, help="path to a csv")
parser.add_argument("--out", type=str, help="dest for edited images")
args = parser.parse_args()
