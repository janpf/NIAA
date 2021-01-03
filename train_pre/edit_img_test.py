from pathlib import Path
import sys

sys.path[0] = "."
from train_pre import preprocess_images

in_f = Path("/home/janpf/projects/NIAA/test_imgs/test.jpg")
out_f: Path = in_f.parent / "out"

out_f.mkdir(parents=True, exist_ok=True)

ed = preprocess_images.ImageEditor()

dists = preprocess_images.mapping["all_changes"]

res = ed.distort_list_image(path=str(in_f), distortion_intens_tuple_list=dists)

for k, v in res.items():
    v.save(out_f / f"{k}{in_f.suffix}")
