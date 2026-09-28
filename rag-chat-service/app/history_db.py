import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag_chat_service.history")

# Store database file in backend/rag-chat-service/data/
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "chat_history.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize conversation and message tables."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user 
                ON conversations(user_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    attachment_text TEXT,
                    has_attachment INTEGER DEFAULT 0,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id, created_at ASC)
                """
            )
    finally:
        conn.close()


# Initialize on import
init_db()


def create_conversation(user_id: int, title: str = "New Conversation", conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new conversation session for the given user."""
    cid = conversation_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, user_id, title, now, now),
            )
        return {
            "id": cid,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
    finally:
        conn.close()


def get_user_conversations(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all conversations for a user ordered by last update."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_conversation(conversation_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get single conversation details if owned by user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Get ordered list of messages in a conversation."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, conversation_id, role, content, attachment_text, has_attachment, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            sources_val = None
            if row["sources"]:
                try:
                    sources_val = json.loads(row["sources"])
                except Exception:
                    sources_val = []
            results.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "attachment_text": row["attachment_text"],
                    "has_attachment": bool(row["has_attachment"]),
                    "sources": sources_val,
                    "created_at": row["created_at"],
                }
            )
        return results
    finally:
        conn.close()


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    attachment_text: Optional[str] = None,
    has_attachment: bool = False,
    sources: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Add a message to a conversation and update the conversation timestamp."""
    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources) if sources else None
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, attachment_text, has_attachment, sources, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    conversation_id,
                    role,
                    content,
                    attachment_text,
                    1 if has_attachment else 0,
                    sources_json,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )
        return {
            "id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "attachment_text": attachment_text,
            "has_attachment": has_attachment,
            "sources": sources,
            "created_at": now,
        }
    finally:
        conn.close()


def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Update title of a conversation."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, now, conversation_id),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_conversation(conversation_id: str, user_id: int) -> bool:
    """Delete conversation and cascade delete all its messages."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                DELETE FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()
