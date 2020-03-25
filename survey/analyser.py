import sqlite3
import redis

r = redis.Redis(host="redis")  # TODO redis memory
# httpagentparser?

# conn = sqlite3.connect("/scratch/stud/pfister/NIAA/pexels/logs/submissions.db")
conn = sqlite3.connect("/data/logs/submissions.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()
sdata = c.execute("""SELECT * FROM submissions ORDER BY id""").fetchall()
usercount = c.execute("""SELECT userid, COUNT(id) FROM submissions GROUP BY userid ORDER BY COUNT(id) DESC""").fetchall()
choicecount = c.execute("""SELECT chosen, COUNT(id) as count FROM submissions GROUP BY chosen ORDER BY chosen""").fetchall()
conn.commit()
conn.close()

print("preprocessing:")
print(f"queued: {r.llen('q')}")
print(f"prepared: {r.llen('pairs')}")
if r.hlen("imgs") / 2 != r.llen("pairs"):
    print(f"imgs:{r.hlen('imgs')}")
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
