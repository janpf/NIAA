import sqlite3

queueDB = "/data/logs/queue.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS queue""")
c.execute("""CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, img text, parameter text, leftChanges text, rightChanges text, hashval text)""")
conn.commit()
conn.close()

queueDB = "/data/logs/submissions.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, img text, parameter text, leftChanges text, rightChanges text, chosen text, hashval text, screenWidth text, screenHeight text, windowWidth text, windowHeight text, colorDepth text, userid text, usersubs integer)""")
conn.commit()
conn.close()
