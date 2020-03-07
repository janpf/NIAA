import sqlite3

queueDB = "/data/logs/queue.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS queue""")
c.execute("""CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, img text, parameter text, leftChanges text, rightChanges text, hashval text)""")
conn.commit()
conn.close()
