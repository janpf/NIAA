import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2


@contextlib.contextmanager
def edited_image(img_path: str, change: str, value: float) -> tempfile.NamedTemporaryFile:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as out: #  delete false, da yield den context verlässt
        if "lcontrast" == change:  # CLAHE
            img = cv2.imread(str(img_path))
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)

            cl = cv2.createCLAHE(clipLimit=value, tileGridSize=(8, 8)).apply(l)

            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite(out.name, img)

        elif "exposure" == change: # [-10, 0, 10]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "exposure", f"{change}={value}"])

        elif "temperature" == change: # [1000, 6500, 12000]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "temperature", f"intended-temperature={value}"])

        elif "hue" == change: # [-180, 0, 180]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "hue-chroma", f"{change}={value}"])

        elif "saturation" == change: # [0, 1, 2]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "saturation", f"scale={value}"])

        elif "brightness" == change or "contrast" == change: # [0, 1, 2]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "brightness-contrast", f"{change}={value}"])

        elif "shadows" == change or "highlights" == change: # [-100, 0, 100]
            subprocess.run(["gegl", "-i", str(img_path), "-o", out.name, "--", "shadows-highlights", f"{change}={value}"])

        try:
            yield out
        finally:
            out.close()
            os.remove(tmp.name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to a single image", default="/scratch/stud/pfister/NIAA/442284.jpg")
    parser.add_argument("--parameter", type=str, help="what to change: brightness, contrast...")
    parser.add_argument("--value", type=float, help="change value")
    parser.add_argument("--out", type=str, help="dest for edited images", default="/scratch/stud/pfister/NIAA/AVA/changed")
    args = parser.parse_args()


    (Path(args.out) / args.parameter).mkdir(parents=True, exist_ok=True)
    outfile = Path(args.out) / args.parameter / f"{Path(args.image).stem}_{args.parameter}_{args.value}.jpg"

    print(outfile)
    with edited_image(img_path=args.image, change=args.parameter, value=args.value) as tmp:
        shutil.copy(tmp.name, outfile)
