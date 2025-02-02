import sqlite3

# SQLite setup: Create the database and table if they don't exist
def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def store_message(session_id: str, role: str, message: str, db_name="conversation_history.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO conversation_history (session_id, role, message) 
    VALUES (?, ?, ?)
    ''', (session_id, role, message))
    conn.commit()
    conn.close()


# Retrieve conversation history for a session, ordered by timestamp
def get_conversation_history(session_id: str, db_name="conversation_history.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT role, message, timestamp 
    FROM conversation_history 
    WHERE session_id = ? 
    ORDER BY timestamp
    ''', (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": message} for role, message, timestamp in messages]