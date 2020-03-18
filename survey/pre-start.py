import sqlite3

queueDB = "/data/logs/queue.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS queue""")
c.execute("""CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, time DATETIME DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT "queued", img TEXT, parameter TEXT, leftChanges TEXT, rightChanges TEXT, hashval TEXT)""")
conn.commit()
conn.close()

queueDB = "/data/logs/submissions.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, time DATETIME DEFAULT CURRENT_TIMESTAMP, img TEXT, parameter TEXT, leftChanges TEXT, rightChanges TEXT, chosen TEXT, hashval TEXT, screenWidth TEXT, screenHeight TEXT, windowWidth TEXT, windowHeight TEXT, colorDepth TEXT, userid TEXT, usersubs INTEGER)"""
)
conn.commit()
conn.close()
