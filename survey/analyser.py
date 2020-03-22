import sqlite3

# httpagentparser?


conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/queue.db")
# conn = sqlite3.connect("/data/logs/queue.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()
qdata = c.execute("""SELECT * FROM queue ORDER BY id""").fetchall()
qstatus = c.execute("""SELECT status, COUNT(id) as count FROM queue GROUP BY status ORDER BY status DESC""").fetchall()
conn.commit()
conn.close()

conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/submissions.db")
# conn = sqlite3.connect("/data/logs/queue.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()
sdata = c.execute("""SELECT * FROM submissions ORDER BY id""").fetchall()
usercount = c.execute("""SELECT userid, COUNT(id) FROM submissions GROUP BY userid ORDER BY COUNT(id) DESC""").fetchall()
choicecount = c.execute("""SELECT chosen, COUNT(id) as count FROM submissions GROUP BY chosen ORDER BY chosen""").fetchall()
conn.commit()
conn.close()

print("preprocessing:")
for row in qstatus:
    print(f"\t{row['status']}: {row['count']}")
print("---")
print()

print(f"{len(sdata)} images compared by {len(usercount)} users")
print("overall distribution:")
for row in choicecount:
    print(f"\t{row['chosen']}: {row['count']}")
print("---")
print()

print("Top 5 leaderboard:")
for row in usercount[-5:]:
    print(f"{row[0]}: {row[1]}")
print("---")
print()

print("3 most recent comparisons:")
for row in sdata[-3:]:
    print(tuple(row))

print("---")
print()
