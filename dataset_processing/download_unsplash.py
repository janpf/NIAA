import pandas as pd
from pathlib import Path
import requests
from multiprocessing.pool import ThreadPool

photos = pd.read_csv("/home/stud/pfister/scratch/NIAA/unsplash/photos.tsv000", sep="\t")


img_folder = Path("/home/stud/pfister/scratch/NIAA/unsplash/images")


def fetch_url(url, id):
    url += "?w=1000&fm=jpg&fit=max"
    img_file = img_folder / (id + ".jpg")
    if not img_file.exists():
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(str(img_file), "wb") as f:
                for chunk in r:
                    f.write(chunk)


tuples = [(row.photo_image_url, row.photo_id) for i, row in photos.iterrows()]

pool = ThreadPool(8)
result = pool.starmap(fetch_url, tuples)
pool.close()
