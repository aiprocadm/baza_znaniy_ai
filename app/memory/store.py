import sqlite3, os, time

class MemoryStore:
    def __init__(self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int):
        self.db_path = db_path
        self.ttl = ttl_days*86400
        self.trigger = summary_trigger
        self.max_tokens = max_tokens
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY,
                user_id TEXT, conv_id TEXT, role TEXT, content TEXT, ts INTEGER
            )""")
            c.commit()

    def record(self, user_id: str, conv_id: str|None, msg: str, ans: str):
        conv_id = conv_id or "default"
        now = int(time.time())
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                      (user_id, conv_id, "user", msg, now))
            c.execute("INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                      (user_id, conv_id, "assistant", ans, now))
            c.commit()

    def load_context(self, user_id: str, conv_id: str|None):
        conv_id = conv_id or "default"
        cutoff = int(time.time()) - self.ttl
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("""SELECT role,content FROM messages
                                WHERE user_id=? AND conv_id=? AND ts>=?
                                ORDER BY id DESC LIMIT 10""",
                             (user_id, conv_id, cutoff)).fetchall()
        rows = rows[::-1]
        text = "\n".join(f"{r}: {t}" for r,t in rows)
        return text[:self.max_tokens*2]
