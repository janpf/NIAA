import logging
import threading
import time
from pathlib import Path
from edit_image import edit_image

out = Path("/data/logs/")


def preprocessImage(name):
    logging.info(f"Thread {name}: starting")

    darktable_dir = f"/tmp/darktable/{name}"

    conn = sqlite3.connect(out / "queue.db", isolation_level="EXCLUSIVE")  # completely locks down database for all other accesses
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    while True:
        try:
            conn = sqlite3.connect(out / "queue.db", isolation_level="EXCLUSIVE")  # completely locks down database for all other accesses
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            data = c.execute("""SELECT * FROM queue WHERE status = queued ORDER BY id LIMIT 1""").fetchone()  # first inserted imagepair
            c.execute("""UPDATE queue SET status = working WHERE id = ?""", (data["id"],))
            conn.commit()
            conn.close()
        except:
            time.sleep(1)
            continue

        edit_image(img_path=data["img"], change=data["parameter"], value=data["leftChanges"], out_path=Path("/tmp/imgs/") / f"{Path(data['img']).stem}_l.jpg", darktable_config=darktable_dir)
        edit_image(img_path=data["img"], change=data["parameter"], value=data["rightChanges"], out_path=Path("/tmp/imgs/") / f"{Path(data['img']).stem}_r.jpg", darktable_config=darktable_dir)

        conn = sqlite3.connect(out / "queue.db", isolation_level="NONE")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""UPDATE queue SET status = done WHERE id = ?""", (data["id"],))
        conn.commit()
        conn.close()

    logging.info(f"Thread {name}: finishing")


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    while threading.activeCount() < 4:  # um gestorbene Threads wieder zu starten
        logging.info("Main    : creating one more thread")
        threading.Thread(target=preprocessImage, args=(threading.activeCount(),)).start()
        logging.info(f"Main    : {threading.activeCount()} Threads active")
        time.sleep(1)