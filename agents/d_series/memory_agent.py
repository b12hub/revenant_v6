# agents/d_series/memory_agent.py
"""
Memory Agent for Revenant Framework
D-Series: Persistence, Recall, and Memory Management
Handles short-term and long-term memory storage with semantic recall capabilities.
"""

import json
import sqlite3
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from core.agent_base import RevenantAgentBase

logger = logging.getLogger(__name__)


class MemoryAgent(RevenantAgentBase):
    """Manages persistence, recall, and memory operations across agent interactions."""

    metadata = {
        "name": "MemoryAgent",
        "version": "1.0.0",
        "series": "d_series",
        "description": "Handles persistence, recall, and memory management across Revenant agents",
        "module": "agents.d_series.memory_agent"
    }

    def __init__(self, db_path: str = "memory_store.db", enable_embeddings: bool = False):
        """
        Initialize MemoryAgent with storage backend.

        Args:
            db_path: Path to SQLite database file
            enable_embeddings: Whether to enable semantic embedding features
        """
        super().__init__()
        self.db_path = Path(db_path)
        self.enable_embeddings = enable_embeddings
        self._lock = asyncio.Lock()

        # Initialize database
        self._init_database()
        logger.info(f"Initialized {self.metadata['series']}-Series MemoryAgent v{self.metadata['version']}")

    async def store_memory(self, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store memory with context identifier and metadata.

        Args:
            context_id: Unique identifier for memory context
            data: Memory data to store

        Returns:
            Operation status and stored memory metadata
        """
        async with self._lock:
            try:
                timestamp = datetime.utcnow().isoformat()
                memory_id = self._generate_memory_id(context_id, data)

                # Create memory entry
                memory_entry = {
                    "memory_id": memory_id,
                    "context_id": context_id,
                    "data": data,
                    "timestamp": timestamp,
                    "access_count": 0,
                    "importance_score": data.get('importance', 0.5),
                    "expires_at": self._calculate_expiration(data.get('ttl_hours', 24))
                }

                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO memories 
                    (memory_id, context_id, data, timestamp, access_count, importance_score, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_entry['memory_id'],
                    memory_entry['context_id'],
                    json.dumps(memory_entry['data']),
                    memory_entry['timestamp'],
                    memory_entry['access_count'],
                    memory_entry['importance_score'],
                    memory_entry['expires_at']
                ))

                conn.commit()
                conn.close()

                logger.info(f"Stored memory {memory_id} for context {context_id}")

                return {
                    "status": "success",
                    "memory_id": memory_id,
                    "context_id": context_id,
                    "timestamp": timestamp,
                    "details": f"Memory stored successfully with ID: {memory_id}"
                }

            except Exception as e:
                logger.error(f"Failed to store memory for context {context_id}: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "details": f"Memory storage failed: {str(e)}"
                }

    async def recall_memory(self, context_id: str, query: Optional[str] = None,
                            limit: int = 10) -> Dict[str, Any]:
        """
        Recall memories by context ID with optional semantic query.

        Args:
            context_id: Context identifier to recall memories for
            query: Optional query string for semantic search
            limit: Maximum number of memories to return

        Returns:
            Retrieved memories with metadata
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if query and self.enable_embeddings:
                    # Semantic recall using embeddings
                    memories = self._semantic_recall(context_id, query, limit)
                else:
                    # Simple context-based recall
                    cursor.execute('''
                        SELECT memory_id, context_id, data, timestamp, access_count, importance_score
                        FROM memories 
                        WHERE context_id = ? AND (expires_at IS NULL OR expires_at > ?)
                        ORDER BY importance_score DESC, timestamp DESC
                        LIMIT ?
                    ''', (context_id, datetime.utcnow().isoformat(), limit))

                    rows = cursor.fetchall()
                    memories = []

                    for row in rows:
                        memory_data = json.loads(row[2])
                        memories.append({
                            "memory_id": row[0],
                            "context_id": row[1],
                            "data": memory_data,
                            "timestamp": row[3],
                            "access_count": row[4],
                            "importance_score": row[5]
                        })

                        # Update access count
                        cursor.execute('''
                            UPDATE memories SET access_count = access_count + 1 
                            WHERE memory_id = ?
                        ''', (row[0],))

                conn.commit()
                conn.close()

                logger.info(f"Recalled {len(memories)} memories for context {context_id}")

                return {
                    "status": "success",
                    "context_id": context_id,
                    "memories": memories,
                    "total_recalled": len(memories),
                    "query_used": query,
                    "details": f"Successfully recalled {len(memories)} memories"
                }

            except Exception as e:
                logger.error(f"Failed to recall memories for context {context_id}: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "memories": [],
                    "details": f"Memory recall failed: {str(e)}"
                }

    async def summarize_history(self, history: List[str]) -> str:
        """
        Generate summary of conversation history.

        Args:
            history: List of conversation turns or messages

        Returns:
            Concise summary of the history
        """
        try:
            if not history:
                return "No history available."

            # Simple summarization for now - can be enhanced with LLM
            if len(history) <= 3:
                return "Brief conversation: " + " | ".join(history[:3])

            # For longer histories, create a structured summary
            key_points = []
            turns_considered = min(5, len(history))

            for i, turn in enumerate(history[:turns_considered]):
                if len(turn) > 100:
                    key_points.append(f"Turn {i + 1}: {turn[:100]}...")
                else:
                    key_points.append(f"Turn {i + 1}: {turn}")

            summary = f"Conversation summary ({len(history)} turns): " + "; ".join(key_points)

            if len(history) > turns_considered:
                summary += f" ... and {len(history) - turns_considered} more turns"

            logger.info(f"Generated summary for {len(history)} history items")
            return summary

        except Exception as e:
            logger.error(f"History summarization failed: {str(e)}")
            return f"Summary generation failed: {str(e)}"

    async def cleanup_expired(self, max_age: int = 168) -> Dict[str, Any]:
        """
        Remove expired memories older than specified age.

        Args:
            max_age: Maximum age in hours for memories to keep

        Returns:
            Cleanup operation results
        """
        async with self._lock:
            try:
                cutoff_time = (datetime.utcnow() - timedelta(hours=max_age)).isoformat()

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Count before deletion for reporting
                cursor.execute('SELECT COUNT(*) FROM memories WHERE timestamp < ?', (cutoff_time,))
                count_before = cursor.fetchone()[0]

                # Delete expired memories
                cursor.execute('DELETE FROM memories WHERE timestamp < ?', (cutoff_time,))
                deleted_count = cursor.rowcount

                conn.commit()
                conn.close()

                logger.info(f"Cleaned up {deleted_count} expired memories (older than {max_age}h)")

                return {
                    "status": "success",
                    "deleted_count": deleted_count,
                    "max_age_hours": max_age,
                    "cutoff_time": cutoff_time,
                    "details": f"Removed {deleted_count} memories older than {max_age} hours"
                }

            except Exception as e:
                logger.error(f"Memory cleanup failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "deleted_count": 0,
                    "details": f"Cleanup failed: {str(e)}"
                }

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM memories')
            total_memories = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT context_id) FROM memories')
            unique_contexts = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(access_count) FROM memories')
            avg_access = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT context_id, COUNT(*) as count 
                FROM memories 
                GROUP BY context_id 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            top_contexts = [{"context_id": row[0], "count": row[1]} for row in cursor.fetchall()]

            conn.close()

            return {
                "status": "success",
                "metrics": {
                    "total_memories": total_memories,
                    "unique_contexts": unique_contexts,
                    "average_access_count": round(avg_access, 2),
                    "top_contexts": top_contexts
                },
                "details": f"Memory store contains {total_memories} memories across {unique_contexts} contexts"
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "metrics": {},
                "details": f"Statistics retrieval failed: {str(e)}"
            }

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5,
                    expires_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_id ON memories (context_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories (importance_score)')

            conn.commit()
            conn.close()
            logger.debug("Memory database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _generate_memory_id(self, context_id: str, data: Dict[str, Any]) -> str:
        """Generate unique memory ID from context and data."""
        content_string = f"{context_id}_{json.dumps(data, sort_keys=True)}"
        return hashlib.md5(content_string.encode()).hexdigest()[:16]

    def _calculate_expiration(self, ttl_hours: int) -> str:
        """Calculate expiration timestamp."""
        return (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()

    def _semantic_recall(self, context_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform semantic recall using embeddings.
        Placeholder for future embedding implementation.
        """
        # This would integrate with sentence transformers or similar
        # For now, fall back to keyword matching
        logger.warning("Embedding features not yet implemented, using fallback recall")
        return []