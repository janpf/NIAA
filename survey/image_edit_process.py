import logging
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")
from edit_image import edit_image


out = Path("/data/logs/")
editedImageFolder = Path("/tmp/imgs/")


def preprocessImage(name):
    logging.info(f"Thread {name}: starting")

    darktable_dir = f"/tmp/darktable/{name}"
    Path(darktable_dir).mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out / "queue.db", isolation_level="EXCLUSIVE")  # completely locks down database for all other accesses
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    while True:
        try:
            conn = sqlite3.connect(out / "queue.db", isolation_level="EXCLUSIVE")  # completely locks down database for all other accesses
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            data = c.execute("""SELECT * FROM queue WHERE status = "queued" ORDER BY id LIMIT 1""").fetchone()  # first inserted imagepair
            c.execute("""UPDATE queue SET status = "working" WHERE id = ?""", (data["id"],))
            conn.commit()
            conn.close()
        except:
            time.sleep(1)
            continue

        edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_l.jpg", darktable_config=darktable_dir)
        edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_r.jpg", darktable_config=darktable_dir)

        conn = sqlite3.connect(out / "queue.db", isolation_level=None)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""UPDATE queue SET status = "done" WHERE id = ?""", (data["id"],))
        conn.commit()
        conn.close()

    logging.info(f"Thread {name}: finishing")


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while True:  # um gestorbene Threads wieder zu starten
        if threading.activeCount() < 5:
            logging.info("Main    : creating one more thread")
            threading.Thread(target=preprocessImage, args=(threading.activeCount(),)).start()
            logging.info(f"Main    : {threading.activeCount()} Threads active")
            subprocess.Popen(f"ls -tp {editedImageFolder} | grep -v '/$' | tail -n +201 | xargs -d '\n' -r rm --", shell=True)  # only keep 200 latest images
            time.sleep(1)
