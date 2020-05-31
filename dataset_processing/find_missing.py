import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
from edit_image import parameter_range

pexels_dir = Path("/scratch") / "stud" / "pfister" / "NIAA" / "pexels"
img_dir = pexels_dir / "images"
out_dir = pexels_dir / "edited_images"

del parameter_range["lcontrast"]
orig_imgs = list(img_dir.iterdir())
orig_imgs: set = {str(img.name) for img in orig_imgs}

missing_all = set(orig_imgs)
missing_sw = set()
for parameter in parameter_range:
    for change in parameter_range[parameter]["range"]:
        if math.isclose(change, parameter_range[parameter]["default"]):
            continue
        edited_imgs = list((out_dir / parameter / str(change)).iterdir())
        edited_imgs: set = {str(img.name) for img in edited_imgs}
        missing = orig_imgs.difference(edited_imgs)
        missing_all.intersection_update(missing)
        missing_sw.update(missing)
        print(parameter, change)
        print(len(missing))

print("missing all:", len(missing_all))
print(missing_all)

print("missing somewhere:", len(missing_sw))
print(missing_sw)
