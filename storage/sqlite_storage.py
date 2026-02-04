"""
存储层：基于SQLite的持久化存储
轻量级，适合嵌入式设备
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.models import (
    Episode, UserProfile, Fact, WorkingMemory, 
    Message, MessageRole
)
from config import MemoryConfig


class SQLiteStorage:
    """SQLite存储引擎"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db_path = config.get_db_path()
        
        # 确保数据目录存在
        os.makedirs(config.data_dir, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 用户画像表
            cursor.execute('''
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
            ''')
            
            # 情景记忆表
            cursor.execute('''
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
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance)')
            
            # 知识事实表
            cursor.execute('''
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
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)')
            
            # 工作记忆表（用于持久化当前会话）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS working_memory (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    messages TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_working_user ON working_memory(user_id)')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ========== 用户画像操作 ==========
    
    def save_user_profile(self, profile: UserProfile):
        """保存用户画像"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, name, age, gender, tags, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.user_id,
                profile.name,
                profile.age,
                profile.gender,
                json.dumps(profile.tags, ensure_ascii=False),
                json.dumps(profile.preferences, ensure_ascii=False),
                profile.created_at.isoformat(),
                profile.updated_at.isoformat()
            ))
            conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if row:
                return UserProfile(
                    user_id=row['user_id'],
                    name=row['name'] or "",
                    age=row['age'],
                    gender=row['gender'] or "",
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    preferences=json.loads(row['preferences']) if row['preferences'] else {},
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
            return None
    
    # ========== 情景记忆操作 ==========
    
    def save_episode(self, episode: Episode):
        """保存情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 检查用户情景数量限制
            cursor.execute(
                'SELECT COUNT(*) FROM episodes WHERE user_id = ?', 
                (episode.user_id,)
            )
            count = cursor.fetchone()[0]
            
            if count >= self.config.max_episodes_per_user:
                # 删除最旧且重要性最低的记忆
                cursor.execute('''
                    DELETE FROM episodes WHERE id IN (
                        SELECT id FROM episodes 
                        WHERE user_id = ? 
                        ORDER BY importance ASC, last_accessed ASC 
                        LIMIT ?
                    )
                ''', (episode.user_id, count - self.config.max_episodes_per_user + 1))
            
            # 插入新记忆
            embedding_blob = None
            if episode.embedding:
                import struct
                embedding_blob = struct.pack(f'{len(episode.embedding)}f', *episode.embedding)
            
            cursor.execute('''
                INSERT OR REPLACE INTO episodes 
                (id, user_id, summary, keywords, emotion, importance, access_count,
                 created_at, last_accessed, source_session_id, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
                embedding_blob
            ))
            conn.commit()
    
    def get_episodes(
        self, 
        user_id: str, 
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Episode]:
        """获取用户的情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM episodes 
                WHERE user_id = ? AND importance >= ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            ''', (user_id, min_importance, limit))
            
            episodes = []
            for row in cursor.fetchall():
                ep = Episode(
                    id=row['id'],
                    user_id=row['user_id'],
                    summary=row['summary'],
                    keywords=json.loads(row['keywords']) if row['keywords'] else [],
                    emotion=row['emotion'] or "",
                    importance=row['importance'],
                    access_count=row['access_count'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_accessed=datetime.fromisoformat(row['last_accessed']),
                    source_session_id=row['source_session_id'] or "",
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                episodes.append(ep)
            
            return episodes
    
    def search_episodes_by_keywords(
        self, 
        user_id: str, 
        keywords: List[str],
        limit: int = 5
    ) -> List[Episode]:
        """通过关键词搜索情景记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 构建关键词匹配条件
            conditions = []
            params = [user_id]
            for kw in keywords:
                conditions.append("(summary LIKE ? OR keywords LIKE ?)")
                params.extend([f'%{kw}%', f'%{kw}%'])
            
            if not conditions:
                return []
            
            query = f'''
                SELECT * FROM episodes 
                WHERE user_id = ? AND ({" OR ".join(conditions)})
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            '''
            params.append(limit)
            
            cursor.execute(query, params)
            
            episodes = []
            for row in cursor.fetchall():
                ep = Episode(
                    id=row['id'],
                    user_id=row['user_id'],
                    summary=row['summary'],
                    keywords=json.loads(row['keywords']) if row['keywords'] else [],
                    emotion=row['emotion'] or "",
                    importance=row['importance'],
                    access_count=row['access_count'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_accessed=datetime.fromisoformat(row['last_accessed']),
                    source_session_id=row['source_session_id'] or "",
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                episodes.append(ep)
            
            return episodes
    
    def update_episode_access(self, episode_id: str):
        """更新情景记忆的访问记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE episodes 
                SET access_count = access_count + 1, 
                    last_accessed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), episode_id))
            conn.commit()
    
    def delete_weak_episodes(self, user_id: str, min_strength: float = 0.2) -> int:
        """删除弱记忆（遗忘机制）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取所有记忆并计算强度
            episodes = self.get_episodes(user_id, limit=1000, min_importance=0)
            weak_ids = []
            
            for ep in episodes:
                strength = ep.calculate_strength(self.config.memory_decay_days)
                if strength < min_strength:
                    weak_ids.append(ep.id)
            
            if weak_ids:
                placeholders = ','.join(['?' for _ in weak_ids])
                cursor.execute(
                    f'DELETE FROM episodes WHERE id IN ({placeholders})', 
                    weak_ids
                )
                conn.commit()
            
            return len(weak_ids)
    
    # ========== 知识事实操作 ==========
    
    def save_fact(self, fact: Fact):
        """保存知识事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 检查是否存在相同的事实（去重）
            cursor.execute('''
                SELECT id FROM facts 
                WHERE user_id = ? AND subject = ? AND predicate = ? AND object = ?
            ''', (fact.user_id, fact.subject, fact.predicate, fact.object))
            
            existing = cursor.fetchone()
            if existing:
                # 更新已存在的事实
                cursor.execute('''
                    UPDATE facts 
                    SET confidence = ?, last_verified = ?
                    WHERE id = ?
                ''', (fact.confidence, datetime.now().isoformat(), existing['id']))
            else:
                # 检查数量限制
                cursor.execute(
                    'SELECT COUNT(*) FROM facts WHERE user_id = ?', 
                    (fact.user_id,)
                )
                count = cursor.fetchone()[0]
                
                if count >= self.config.max_facts_per_user:
                    # 删除置信度最低的事实
                    cursor.execute('''
                        DELETE FROM facts WHERE id IN (
                            SELECT id FROM facts 
                            WHERE user_id = ? 
                            ORDER BY confidence ASC 
                            LIMIT ?
                        )
                    ''', (fact.user_id, count - self.config.max_facts_per_user + 1))
                
                cursor.execute('''
                    INSERT INTO facts 
                    (id, user_id, subject, predicate, object, confidence, source, created_at, last_verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fact.id,
                    fact.user_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.confidence,
                    fact.source,
                    fact.created_at.isoformat(),
                    fact.last_verified.isoformat()
                ))
            
            conn.commit()
    
    def get_facts(self, user_id: str, limit: int = 20) -> List[Fact]:
        """获取用户的知识事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM facts 
                WHERE user_id = ?
                ORDER BY confidence DESC, last_verified DESC
                LIMIT ?
            ''', (user_id, limit))
            
            facts = []
            for row in cursor.fetchall():
                fact = Fact(
                    id=row['id'],
                    user_id=row['user_id'],
                    subject=row['subject'],
                    predicate=row['predicate'],
                    object=row['object'],
                    confidence=row['confidence'],
                    source=row['source'] or "",
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_verified=datetime.fromisoformat(row['last_verified'])
                )
                facts.append(fact)
            
            return facts
    
    def search_facts(
        self, 
        user_id: str, 
        query: str,
        limit: int = 10
    ) -> List[Fact]:
        """搜索相关知识事实"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM facts 
                WHERE user_id = ? AND (
                    subject LIKE ? OR predicate LIKE ? OR object LIKE ?
                )
                ORDER BY confidence DESC
                LIMIT ?
            ''', (user_id, f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            facts = []
            for row in cursor.fetchall():
                fact = Fact(
                    id=row['id'],
                    user_id=row['user_id'],
                    subject=row['subject'],
                    predicate=row['predicate'],
                    object=row['object'],
                    confidence=row['confidence'],
                    source=row['source'] or "",
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_verified=datetime.fromisoformat(row['last_verified'])
                )
                facts.append(fact)
            
            return facts
    
    # ========== 工作记忆操作 ==========
    
    def save_working_memory(self, memory: WorkingMemory):
        """保存工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO working_memory 
                (session_id, user_id, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                memory.session_id,
                memory.user_id,
                json.dumps([m.to_dict() for m in memory.messages], ensure_ascii=False),
                memory.created_at.isoformat(),
                memory.updated_at.isoformat()
            ))
            conn.commit()
    
    def get_working_memory(self, session_id: str) -> Optional[WorkingMemory]:
        """获取工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM working_memory WHERE session_id = ?', 
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                messages_data = json.loads(row['messages']) if row['messages'] else []
                messages = [Message.from_dict(m) for m in messages_data]
                
                return WorkingMemory(
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    messages=messages,
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
            return None
    
    def delete_working_memory(self, session_id: str):
        """删除工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM working_memory WHERE session_id = ?', 
                (session_id,)
            )
            conn.commit()
    
    def cleanup_old_sessions(self, days: int = 7) -> int:
        """清理旧的工作记忆"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            from datetime import timedelta
            cutoff = cutoff - timedelta(days=days)
            
            cursor.execute(
                'DELETE FROM working_memory WHERE updated_at < ?',
                (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted
    
    # ========== 统计信息 ==========
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {"user_id": user_id}
            
            # 情景记忆数量
            cursor.execute(
                'SELECT COUNT(*) FROM episodes WHERE user_id = ?', 
                (user_id,)
            )
            stats["episode_count"] = cursor.fetchone()[0]
            
            # 知识事实数量
            cursor.execute(
                'SELECT COUNT(*) FROM facts WHERE user_id = ?', 
                (user_id,)
            )
            stats["fact_count"] = cursor.fetchone()[0]
            
            # 是否有用户画像
            cursor.execute(
                'SELECT COUNT(*) FROM user_profiles WHERE user_id = ?', 
                (user_id,)
            )
            stats["has_profile"] = cursor.fetchone()[0] > 0
            
            return stats
