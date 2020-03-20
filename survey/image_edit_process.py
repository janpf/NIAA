import logging
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import SimpleQueue

sys.path.insert(0, ".")
from edit_image import edit_image



queueDB = "/data/logs/queue.db"
editedImageFolder = Path("/tmp/imgs/")


def preprocessImage(name: int, q: SimpleQueue):
    logging.info(f"Thread {name}: starting")

    darktable_dir = f"/tmp/darktable/{name}"
    Path(darktable_dir).mkdir(parents=True, exist_ok=True)

    while True:
        data = q.get()
        print(f"Thread {name} received {data}")
        edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_l.jpg", darktable_config=darktable_dir)
        edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_r.jpg", darktable_config=darktable_dir)

        conn = sqlite3.connect(queueDB, isolation_level=None)
        c = conn.cursor()
        c.execute("""UPDATE queue SET status = "done" WHERE id = ?""", (data["id"],))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    q = SimpleQueue()
    conn = sqlite3.connect(queueDB, isolation_level=None)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    while True:  # um gestorbene Threads wieder zu starten
        if threading.activeCount() < 6:
            logging.info("Main    : creating one more thread")
            threading.Thread(target=preprocessImage, args=(threading.activeCount(), q)).start()
            logging.info(f"Main    : {threading.activeCount()-1} Threads active")

        try:
            subprocess.Popen(f"ls -tp {editedImageFolder} | grep -v '/$' | tail -n +201 | xargs -d '\n' -r rm --", shell=True)  # only keep 200 latest images
            data = c.execute("""SELECT * FROM queue WHERE status = "queued" ORDER BY id LIMIT 1""").fetchone()  # first inserted imagepair
            c.execute("""UPDATE queue SET status = "working" WHERE id = ?""", (data["id"],))
            conn.commit()
            print("queuing", data)
            q.put(data)
        except Exception as e:
            time.sleep(1)
