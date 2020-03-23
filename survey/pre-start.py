import sqlite3
import redis

r = redis.Redis(host="survey-redis")
r.flushall()

queueDB = "/data/logs/submissions.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, time DATETIME DEFAULT CURRENT_TIMESTAMP, loadTime DATETIME, img TEXT, parameter TEXT, leftChanges TEXT, rightChanges TEXT, chosen TEXT, hashval TEXT, screenWidth TEXT, screenHeight TEXT, windowWidth TEXT, windowHeight TEXT, colorDepth TEXT, userid TEXT, usersubs INTEGER, useragent TEXT)"""
)
conn.commit()
conn.close()
