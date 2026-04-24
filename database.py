import sqlite3
import os

DB_PATH = "fake_news_system.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for source credibility
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT UNIQUE,
            trust_score REAL DEFAULT 50.0,
            verified INTEGER DEFAULT 0
        )
    ''')
    
    # Table for user feedback
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT,
            predicted_label TEXT,
            user_label TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for historical patterns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            frequency INTEGER DEFAULT 1,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Seed some data
    cursor.execute("INSERT OR IGNORE INTO sources (source_name, trust_score, verified) VALUES ('BBC', 95.0, 1)")
    cursor.execute("INSERT OR IGNORE INTO sources (source_name, trust_score, verified) VALUES ('Reuters', 98.0, 1)")
    cursor.execute("INSERT OR IGNORE INTO sources (source_name, trust_score, verified) VALUES ('The Hindu', 90.0, 1)")
    cursor.execute("INSERT OR IGNORE INTO sources (source_name, trust_score, verified) VALUES ('Daily Mail', 40.0, 0)")
    cursor.execute("INSERT OR IGNORE INTO sources (source_name, trust_score, verified) VALUES ('WhatsApp Forward', 10.0, 0)")
    
    conn.commit()
    conn.close()

def save_feedback(text, predicted, user):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (news_text, predicted_label, user_label) VALUES (?, ?, ?)", (text, predicted, user))
    conn.commit()
    conn.close()

def get_source_score(source_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT trust_score FROM sources WHERE source_name = ?", (source_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 50.0

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
