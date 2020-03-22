import sqlite3


conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/queue.db", isolation_level=None)
# conn = sqlite3.connect("/data/logs/queue.db", isolation_level=None)
conn.row_factory = sqlite3.Row
c = conn.cursor()
qdata = c.execute("""SELECT * FROM queue ORDER BY id""").fetchall()
conn.commit()
conn.close()

conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/submissions.db", isolation_level=None)
# conn = sqlite3.connect("/data/logs/queue.db", isolation_level=None)
conn.row_factory = sqlite3.Row
c = conn.cursor()
sdata = c.execute("""SELECT * FROM submissions ORDER BY id""").fetchall()
conn.commit()
conn.close()

queued = sum([1 for row in qdata if row["status"] == "queued"])
working = sum([1 for row in qdata if row["status"] == "working"])
done = sum([1 for row in qdata if row["status"] == "done"])

print(f"preprocessing: done: {done} | working: {working} | queued: {queued}")
print()
print(f"{len(sdata)} images compared")

print("5 most recent comparisons:")
for row in sdata[-5:]:
    print(tuple(row))
