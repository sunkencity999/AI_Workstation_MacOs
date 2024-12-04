import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class DatabaseManager:
    def __init__(self, db_path: str = "conversations.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    title TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            conn.commit()

    def create_conversation(self, model_name: str, title: Optional[str] = None, 
                          tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> int:
        """Create a new conversation and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (model_name, title, tags, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                model_name,
                title,
                json.dumps(tags) if tags else None,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str, 
                   metadata: Optional[Dict] = None) -> int:
        """Add a message to a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO messages (conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                conversation_id,
                role,
                content,
                json.dumps(metadata) if metadata else None
            ))
            
            # Update conversation last_updated timestamp
            cursor.execute('''
                UPDATE conversations
                SET last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (conversation_id,))
            
            conn.commit()
            return cursor.lastrowid

    def get_conversation(self, conversation_id: int) -> Tuple[Dict, List[Dict]]:
        """Get a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get conversation details
            cursor.execute('''
                SELECT * FROM conversations WHERE id = ?
            ''', (conversation_id,))
            conversation = dict(cursor.fetchone())
            
            # Get all messages
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))
            messages = [dict(row) for row in cursor.fetchall()]
            
            return conversation, messages

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List recent conversations with messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get conversations with basic metadata
            cursor.execute('''
                SELECT 
                    c.*,
                    COUNT(m.id) as message_count,
                    MIN(m.timestamp) as first_message,
                    MAX(m.timestamp) as last_message
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.last_updated DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            conversations = [dict(row) for row in cursor.fetchall()]
            
            # Get messages for each conversation
            for conv in conversations:
                cursor.execute('''
                    SELECT * FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp
                ''', (conv['id'],))
                conv['messages'] = [dict(row) for row in cursor.fetchall()]
            
            return conversations

    def search_conversations(self, query: str, limit: int = 50) -> List[Dict]:
        """Search conversations and messages for specific text."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # First get all conversations that have matching content
            cursor.execute('''
                SELECT DISTINCT
                    c.*,
                    COUNT(m.id) as message_count,
                    MIN(m.timestamp) as first_message,
                    MAX(m.timestamp) as last_message
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE 
                    c.title LIKE ? OR
                    c.tags LIKE ? OR
                    m.content LIKE ?
                GROUP BY c.id
                ORDER BY c.last_updated DESC
                LIMIT ?
            ''', (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            conversations = [dict(row) for row in cursor.fetchall()]
            
            # For each conversation, only get messages that match the search query
            for conv in conversations:
                cursor.execute('''
                    SELECT * FROM messages 
                    WHERE conversation_id = ? AND (
                        content LIKE ? OR
                        ? IN (
                            SELECT title FROM conversations 
                            WHERE id = messages.conversation_id
                        )
                    )
                    ORDER BY timestamp
                ''', (conv['id'], f"%{query}%", query))
                conv['messages'] = [dict(row) for row in cursor.fetchall()]
            
            # Only return conversations that have matching messages
            return [conv for conv in conversations if conv['messages']]

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation and all its messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Delete messages first due to foreign key constraint
                cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
                cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error deleting conversation: {str(e)}")
            return False

    def export_conversation(self, conversation_id: int, format: str = "json") -> str:
        """Export a conversation in the specified format."""
        conversation, messages = self.get_conversation(conversation_id)
        
        if format == "json":
            return json.dumps({
                "conversation": conversation,
                "messages": messages
            }, indent=2)
        elif format == "markdown":
            md = f"# Conversation {conversation['title'] or conversation['id']}\n\n"
            md += f"Model: {conversation['model_name']}\n"
            md += f"Started: {conversation['start_time']}\n\n"
            
            for msg in messages:
                md += f"## {msg['role'].title()}\n"
                md += f"{msg['content']}\n\n"
            
            return md
        else:
            raise ValueError(f"Unsupported export format: {format}")
