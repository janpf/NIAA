"""
Obsolete as every change made to all images are already saved in 'edited_images'.
Only needed if not all images have been preprocessed or 'queryNIMAsurvey.py' isn't adapted to use those images yet.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from edit_image import edit_image


editedImageFolder = Path("/data") / "surveyimgs"
inputFolder = Path("/data") / "images"


def preprocessImage(img: str, parameter: str, leftChange: float, rightChange: float, hashval: int, outdir: Path = editedImageFolder):
    logging.info(f"working on {img}\t{parameter}\t{leftChange}\t{rightChange}\t{hashval}")
    savelocation = outdir / f"{hashval}l.jpg"
    if savelocation.exists() and not args.redo_existing:
        logging.info(f"skipping:\t{img}\t{parameter}\t{leftChange}\t{hashval}")
    else:
        left = edit_image(img_path=img, change=parameter, value=leftChange)
        left.save(savelocation, format="JPEG")

    savelocation = outdir / f"{hashval}r.jpg"
    if savelocation.exists() and not args.redo_existing:
        logging.info(f"skipping:\t{img}\t{parameter}\t{rightChange}\t{hashval}")
    else:
        right = edit_image(img_path=img, change=parameter, value=rightChange)
        right.save(outdir / f"{hashval}r.jpg", format="JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--redo_existing", action="store_true")
    args = parser.parse_args()

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    df = pd.read_csv(args.csv_file)
    df = df[df.chosen != "error"]
    df = df[df.chosen != "unsure"]

    df = df.iloc[np.random.permutation(len(df))]  #  shuffle df for parallelism
    df = df.reset_index(drop=True)

    logging.info(f"{len(df)} pairs have to get preprocessed")
    for _, row in df.iterrows():
        img = row["img"].replace("/img/", "")
        img = str(inputFolder / img)
        try:
            preprocessImage(img, row["parameter"], row["leftChanges"], row["rightChanges"], row["hashval"])
        except Exception as e:
            logging.info(f"got Exception: {e}")
