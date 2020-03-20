import sqlite3


conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/queue.db", isolation_level=None)
# conn = sqlite3.connect("/data/logs/queue.db", isolation_level=None)
conn.row_factory = sqlite3.Row
c = conn.cursor()
data = c.execute("""SELECT * FROM queue ORDER BY id""").fetchall()
conn.commit()
conn.close()
for row in data:
    print(row)
