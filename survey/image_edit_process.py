import logging
import sqlite3
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
    conn = sqlite3.connect(queueDB)
    c = conn.cursor()

    while True:
        data = q.get()
        print(f"Thread {name} received {data}")
        edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_l.jpg", darktable_config=darktable_dir)
        edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], out_path=editedImageFolder / f"{Path(data['img']).stem}_r.jpg", darktable_config=darktable_dir)
        try:
            c.execute("""UPDATE queue SET status = "done" WHERE id = ?""", (data["id"],))
            conn.commit()
        except:
            conn.close()
            break


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    q = SimpleQueue()
    conn = sqlite3.connect(queueDB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    while True:  # um gestorbene Threads wieder zu starten
        if threading.activeCount() < 6:
            logging.info("Main    : creating one more thread")
            threading.Thread(target=preprocessImage, args=(threading.activeCount(), q)).start()
            logging.info(f"Main    : {threading.activeCount()-1} Threads active")

        try:
            data = c.execute("""SELECT * FROM queue WHERE status = "queued" ORDER BY id""").fetchall()
            c.executemany("""UPDATE queue SET status = "working" WHERE id = ?""", [(row["id"],) for row in data])
            conn.commit()
            for row in data:
                print("queuing", row)
                q.put(row)
        except Exception as e:
            time.sleep(1)
