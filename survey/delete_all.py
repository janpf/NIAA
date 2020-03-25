import sqlite3
import redis

r = redis.Redis(host="redis")
r.flushall()

queueDB = "/data/logs/submissions.db"
conn = sqlite3.connect(queueDB)
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS submissions""")
conn.commit()
conn.close()
