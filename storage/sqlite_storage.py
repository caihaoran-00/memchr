"""
存储层：基于 SQLite 的持久化存储
"""

import json
import os
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig
from memory_core.models import Episode, Fact, Message, UserProfile, WorkingMemory


class SQLiteStorage:
    """SQLite 存储引擎"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db_path = config.get_db_path()
        os.makedirs(config.data_dir, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    tags TEXT,
                    preferences TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    summary TEXT,
                    keywords TEXT,
                    emotion TEXT,
                    importance REAL,
                    access_count INTEGER,
                    created_at TEXT,
                    last_accessed TEXT,
                    source_session_id TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance)"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    confidence REAL,
                    source TEXT,
                    created_at TEXT,
                    last_verified TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memory (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    messages TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_working_user ON working_memory(user_id)"
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_user_profile(self, profile: UserProfile):
        """保存用户画像"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_profiles
                (user_id, name, age, gender, tags, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.user_id,
                    profile.name,
                    profile.age,
                    profile.gender,
                    json.dumps(profile.tags, ensure_ascii=False),
                    json.dumps(profile.preferences, ensure_ascii=False),
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None

            return UserProfile(
                user_id=row["user_id"],
                name=row["name"] or "",
                age=row["age"],
                gender=row["gender"] or "",
                tags=json.loads(row["tags"]) if row["tags"] else [],
                preferences=json.loads(row["preferences"]) if row["preferences"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def save_episode(self, episode: Episode):
        """保存情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE user_id = ?", (episode.user_id,))
            count = cursor.fetchone()[0]

            if count >= self.config.max_episodes_per_user:
                cursor.execute(
                    """
                    DELETE FROM episodes WHERE id IN (
                        SELECT id FROM episodes
                        WHERE user_id = ?
                        ORDER BY importance ASC, last_accessed ASC
                        LIMIT ?
                    )
                    """,
                    (episode.user_id, count - self.config.max_episodes_per_user + 1),
                )

            embedding_blob = None
            if episode.embedding:
                embedding_blob = struct.pack(f"{len(episode.embedding)}f", *episode.embedding)

            cursor.execute(
                """
                INSERT OR REPLACE INTO episodes
                (id, user_id, summary, keywords, emotion, importance, access_count,
                 created_at, last_accessed, source_session_id, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.user_id,
                    episode.summary,
                    json.dumps(episode.keywords, ensure_ascii=False),
                    episode.emotion,
                    episode.importance,
                    episode.access_count,
                    episode.created_at.isoformat(),
                    episode.last_accessed.isoformat(),
                    episode.source_session_id,
                    json.dumps(episode.metadata, ensure_ascii=False),
                    embedding_blob,
                ),
            )
            conn.commit()

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """将数据库行转换为 Episode"""
        embedding = None
        if row["embedding"]:
            count = len(row["embedding"]) // 4
            embedding = list(struct.unpack(f"{count}f", row["embedding"]))

        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            summary=row["summary"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            emotion=row["emotion"] or "",
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            source_session_id=row["source_session_id"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=embedding,
        )

    def get_episodes(
        self, user_id: str, limit: int = 10, min_importance: float = 0.0
    ) -> List[Episode]:
        """获取用户的情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM episodes
                WHERE user_id = ? AND importance >= ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
                """,
                (user_id, min_importance, limit),
            )
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episode_candidates(
        self, user_id: str, limit: Optional[int] = None, min_importance: float = 0.0
    ) -> List[Episode]:
        """获取用于语义检索的候选 Episode"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM episodes
                WHERE user_id = ? AND importance >= ?
                ORDER BY last_accessed DESC, importance DESC
            """
            params: List[Any] = [user_id, min_importance]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def search_episodes_by_keywords(
        self, user_id: str, keywords: List[str], limit: int = 5
    ) -> List[Episode]:
        """通过关键词搜索情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            conditions = []
            params: List[Any] = [user_id]
            for keyword in keywords:
                conditions.append("(summary LIKE ? OR keywords LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])

            if not conditions:
                return []

            query = f"""
                SELECT * FROM episodes
                WHERE user_id = ? AND ({" OR ".join(conditions)})
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """
            params.append(limit)
            cursor.execute(query, params)
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def update_episode_access(self, episode_id: str):
        """更新情景记忆访问记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE episodes
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), episode_id),
            )
            conn.commit()

    def delete_weak_episodes(self, user_id: str, min_strength: float = 0.2) -> int:
        """删除弱记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            episodes = self.get_episodes(user_id, limit=1000, min_importance=0)
            weak_ids = [
                episode.id
                for episode in episodes
                if episode.calculate_strength(self.config.memory_decay_days) < min_strength
            ]

            if weak_ids:
                placeholders = ",".join(["?" for _ in weak_ids])
                cursor.execute(
                    f"DELETE FROM episodes WHERE id IN ({placeholders})", weak_ids
                )
                conn.commit()

            return len(weak_ids)

    def save_fact(self, fact: Fact):
        """保存知识事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM facts
                WHERE user_id = ? AND subject = ? AND predicate = ? AND object = ?
                """,
                (fact.user_id, fact.subject, fact.predicate, fact.object),
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE facts
                    SET confidence = ?, last_verified = ?
                    WHERE id = ?
                    """,
                    (fact.confidence, datetime.now().isoformat(), existing["id"]),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM facts WHERE user_id = ?", (fact.user_id,))
                count = cursor.fetchone()[0]
                if count >= self.config.max_facts_per_user:
                    cursor.execute(
                        """
                        DELETE FROM facts WHERE id IN (
                            SELECT id FROM facts
                            WHERE user_id = ?
                            ORDER BY confidence ASC
                            LIMIT ?
                        )
                        """,
                        (fact.user_id, count - self.config.max_facts_per_user + 1),
                    )
                cursor.execute(
                    """
                    INSERT INTO facts
                    (id, user_id, subject, predicate, object, confidence, source, created_at, last_verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact.id,
                        fact.user_id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        fact.confidence,
                        fact.source,
                        fact.created_at.isoformat(),
                        fact.last_verified.isoformat(),
                    ),
                )
            conn.commit()

    def get_facts(self, user_id: str, limit: int = 20) -> List[Fact]:
        """获取用户事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM facts
                WHERE user_id = ?
                ORDER BY confidence DESC, last_verified DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """将数据库行转换为 Fact"""
        return Fact(
            id=row["id"],
            user_id=row["user_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            source=row["source"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            last_verified=datetime.fromisoformat(row["last_verified"]),
        )

    def search_facts(self, user_id: str, query: str, limit: int = 10) -> List[Fact]:
        """搜索相关知识事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM facts
                WHERE user_id = ? AND (
                    subject LIKE ? OR predicate LIKE ? OR object LIKE ?
                )
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (user_id, f"%{query}%", f"%{query}%", f"%{query}%", limit),
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def save_working_memory(self, memory: WorkingMemory):
        """保存工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO working_memory
                (session_id, user_id, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory.session_id,
                    memory.user_id,
                    json.dumps([message.to_dict() for message in memory.messages], ensure_ascii=False),
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get_working_memory(self, session_id: str) -> Optional[WorkingMemory]:
        """获取工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM working_memory WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            messages_data = json.loads(row["messages"]) if row["messages"] else []
            messages = [Message.from_dict(item) for item in messages_data]
            return WorkingMemory(
                user_id=row["user_id"],
                session_id=row["session_id"],
                messages=messages,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def delete_working_memory(self, session_id: str):
        """删除工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM working_memory WHERE session_id = ?", (session_id,))
            conn.commit()

    def cleanup_old_sessions(self, days: int = 7) -> int:
        """清理过期工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff = cutoff - timedelta(days=days)
            cursor.execute(
                "DELETE FROM working_memory WHERE updated_at < ?", (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {"user_id": user_id}

            cursor.execute("SELECT COUNT(*) FROM episodes WHERE user_id = ?", (user_id,))
            stats["episode_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM facts WHERE user_id = ?", (user_id,))
            stats["fact_count"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            stats["has_profile"] = cursor.fetchone()[0] > 0

            return stats
