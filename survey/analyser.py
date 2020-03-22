import sqlite3


conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/queue.db")
# conn = sqlite3.connect("/data/logs/queue.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()
qdata = c.execute("""SELECT * FROM queue ORDER BY id""").fetchall()
conn.commit()
conn.close()

conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/submissions.db")
# conn = sqlite3.connect("/data/logs/queue.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()
sdata = c.execute("""SELECT * FROM submissions ORDER BY id""").fetchall()
conn.commit()
conn.close()

queued = sum([1 for row in qdata if row["status"] == "queued"])
working = sum([1 for row in qdata if row["status"] == "working"])
done = sum([1 for row in qdata if row["status"] == "done"])
others = sum([1 for row in qdata if row["status"] != "queued" and row["status"] != "working" and row["status"] != "done"])

print(f"preprocessing: done: {done} | working: {working} | queued: {queued} | others: {others}")
print()
print(f"{len(sdata)} images compared")

print("3 most recent comparisons:")
for row in sdata[-3:]:
    print(tuple(row))
