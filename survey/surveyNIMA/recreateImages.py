import argparse
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from edit_image import edit_image


editedImageFolder = Path("/data") / "surveyimgs"
inputFolder = Path("/data") / "images"


def preprocessImage(img: str, parameter: str, leftChange: float, rightChange: float, hashval: int, outdir: Path = editedImageFolder):

    left = edit_image(img_path=img, change=parameter, value=leftChange)
    right = edit_image(img_path=img, change=parameter, value=rightChange)

    if not left or not right:
        return

    left.save(outdir / f"{hashval}l", format="JPEG")
    right.save(outdir / f"{hashval}r", format="JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--redo_existing", action="store_true", type=bool)
    args = parser.parse_args()

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    df = pd.read_csv(args.csv_file)
    df = df[df.chosen != "error"]
    df = df[df.chosen != "unsure"]

    print(f"{len(df)} pairs have to get preprocessed")
    for row in df.iterrows():
        img = row["img"].replace("/img/", "")
        img = inputFolder / img
        if img.exists():
            if args.redo_existing:
                print(f"redoing:\t{img}")
            else:
                print(f"skipping:\t{img}")
        else:
            print(f"working:\t{img}")
            try:
                preprocessImage(img, row["parameter"], row["leftChanges"], row["rightChanges"], row["hashval"])
            except Exception as e:
                print(f"got Exception: {e}")
