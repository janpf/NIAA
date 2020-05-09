import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from edit_image import create_xmp_file, parameter_range

del parameter_range["lcontrast"]
for parameter in parameter_range:
    print()
    out_dir = Path("/scratch") / "stud" / "pfister" / "NIAA" / "pexels" / "xmps" / parameter
    out_dir.mkdir(parents=True, exist_ok=True)

    for val in parameter_range[parameter]["range"]:
        if math.isclose(val, parameter_range[parameter]["default"]):
            print(f"skipping default {val} for {parameter}")
            continue
        print(f"creating to {str(out_dir / f'{val}.xmp')}")
        create_xmp_file(str(out_dir / f"{val}.xmp"), parameter, val)
