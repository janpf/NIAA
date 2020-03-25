import sqlite3

queueDB = "/data/logs/submissions.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, loadTime DATETIME, submitTime DATETIME, img TEXT, parameter TEXT, leftChanges TEXT, rightChanges TEXT, chosen TEXT, hashval TEXT, screenWidth TEXT, screenHeight TEXT, windowWidth TEXT, windowHeight TEXT, colorDepth TEXT, userid TEXT, usersubs INTEGER, useragent TEXT)"""
)
conn.commit()
conn.close()
