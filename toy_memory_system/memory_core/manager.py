"""
记忆管理器：核心控制层
负责记忆的存储、检索、压缩和遗忘
"""
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.models import (
    WorkingMemory, Episode, Fact, UserProfile,
    Message, MessageRole, MemoryContext
)
from memory_core.extractor import MemoryExtractor, RuleBasedExtractor, create_extractor
from memory_core.llm_client import create_llm_client, LLMClient
from storage.sqlite_storage import SQLiteStorage
from config import MemoryConfig


class MemoryManager:
    """记忆管理器：整合记忆系统的核心组件"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # 初始化存储
        self.storage = SQLiteStorage(self.config)
        
        # 初始化LLM客户端
        self.llm_client = create_llm_client(self.config)
        
        # 初始化提取器
        self.extractor = create_extractor(
            self.config, 
            use_llm=(self.config.llm_provider != "mock"),
            llm_client=self.llm_client
        )
        
        # 工作记忆缓存（内存中）
        self._working_memory_cache: Dict[str, WorkingMemory] = {}
    
    # ========== 会话管理 ==========
    
    def start_session(self, user_id: str, session_id: str = None) -> WorkingMemory:
        """开始新会话"""
        session_id = session_id or str(uuid.uuid4())
        
        # 检查是否有未完成的会话
        existing = self.storage.get_working_memory(session_id)
        if existing:
            self._working_memory_cache[session_id] = existing
            return existing
        
        # 创建新的工作记忆
        working_memory = WorkingMemory(
            user_id=user_id,
            session_id=session_id
        )
        self._working_memory_cache[session_id] = working_memory
        self.storage.save_working_memory(working_memory)
        
        return working_memory
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Dict = None
    ):
        """添加消息到工作记忆"""
        if session_id not in self._working_memory_cache:
            working_memory = self.storage.get_working_memory(session_id)
            if not working_memory:
                raise ValueError(f"会话不存在: {session_id}")
            self._working_memory_cache[session_id] = working_memory
        
        working_memory = self._working_memory_cache[session_id]
        msg_role = MessageRole(role) if isinstance(role, str) else role
        working_memory.add_message(msg_role, content, metadata or {})
        
        # 限制工作记忆大小
        max_messages = self.config.working_memory_size * 2  # 每轮2条消息
        if len(working_memory.messages) > max_messages:
            working_memory.messages = working_memory.messages[-max_messages:]
        
        # 持久化
        self.storage.save_working_memory(working_memory)
    
    async def end_session(self, session_id: str, extract_memory: bool = True) -> Optional[Episode]:
        """结束会话，提取记忆"""
        if session_id not in self._working_memory_cache:
            working_memory = self.storage.get_working_memory(session_id)
            if not working_memory:
                return None
        else:
            working_memory = self._working_memory_cache[session_id]
        
        episode = None
        
        # 如果对话足够长，提取记忆
        if extract_memory and len(working_memory.messages) >= self.config.episode_compress_threshold * 2:
            episode = await self._extract_and_store_memory(working_memory)
        
        # 清理工作记忆
        if session_id in self._working_memory_cache:
            del self._working_memory_cache[session_id]
        self.storage.delete_working_memory(session_id)
        
        return episode
    
    async def _extract_and_store_memory(self, working_memory: WorkingMemory) -> Episode:
        """提取并存储记忆"""
        user_id = working_memory.user_id
        session_id = working_memory.session_id
        
        # 使用提取器
        if isinstance(self.extractor, MemoryExtractor):
            extraction = await self.extractor.extract_from_conversation(
                working_memory.messages,
                user_id,
                session_id
            )
        else:
            extraction = self.extractor.extract_from_conversation(
                working_memory.messages,
                user_id,
                session_id
            )
        
        # 创建并保存情景记忆
        episode = self.extractor.create_episode_from_extraction(
            extraction, user_id, session_id
        ) if isinstance(self.extractor, MemoryExtractor) else Episode(
            user_id=user_id,
            summary=extraction["summary"],
            keywords=extraction["keywords"],
            emotion=extraction["emotion"],
            importance=extraction["importance"],
            source_session_id=session_id
        )
        self.storage.save_episode(episode)
        
        # 保存知识事实
        if isinstance(self.extractor, MemoryExtractor):
            facts = self.extractor.create_facts_from_extraction(
                extraction, user_id, session_id
            )
        else:
            facts = [
                Fact(
                    user_id=user_id,
                    subject=f["subject"],
                    predicate=f["predicate"],
                    object=f["object"],
                    source=session_id
                )
                for f in extraction.get("facts", [])
            ]
        
        for fact in facts:
            self.storage.save_fact(fact)
        
        # 更新用户画像
        profile = self.storage.get_user_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
        
        if isinstance(self.extractor, MemoryExtractor):
            profile = self.extractor.update_profile_from_extraction(profile, extraction)
        else:
            updates = extraction.get("profile_updates", {})
            if updates.get("name"):
                profile.name = updates["name"]
            if updates.get("age"):
                profile.age = updates["age"]
            for tag in updates.get("tags", []):
                profile.add_tag(tag)
        
        self.storage.save_user_profile(profile)
        
        return episode
    
    # ========== 记忆检索 ==========
    
    def get_memory_context(
        self, 
        session_id: str, 
        query: str = None
    ) -> MemoryContext:
        """获取记忆上下文（用于增强LLM对话）"""
        # 获取工作记忆
        working_memory = self._working_memory_cache.get(session_id)
        if not working_memory:
            working_memory = self.storage.get_working_memory(session_id)
        
        if not working_memory:
            return MemoryContext()
        
        user_id = working_memory.user_id
        
        # 获取用户画像
        profile = self.storage.get_user_profile(user_id)
        
        # 检索相关情景记忆
        if query:
            keywords = self._extract_query_keywords(query)
            episodes = self.storage.search_episodes_by_keywords(
                user_id, keywords, limit=3
            )
        else:
            episodes = self.storage.get_episodes(
                user_id, limit=3, min_importance=0.5
            )
        
        # 更新访问记录
        for ep in episodes:
            self.storage.update_episode_access(ep.id)
        
        # 检索相关事实
        if query:
            facts = self.storage.search_facts(user_id, query, limit=5)
        else:
            facts = self.storage.get_facts(user_id, limit=5)
        
        return MemoryContext(
            working_memory=working_memory,
            relevant_episodes=episodes,
            user_profile=profile,
            relevant_facts=facts
        )
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        try:
            import jieba
            words = list(jieba.cut(query))
            # 过滤停用词
            stopwords = {"的", "了", "是", "我", "你", "吗", "啊", "呢", "吧", "嘛", "哦", "呀", "什么", "怎么"}
            return [w for w in words if len(w) >= 2 and w not in stopwords][:5]
        except ImportError:
            # jieba未安装，简单分词
            return [query[i:i+2] for i in range(0, min(len(query), 10), 2)]
    
    # ========== 用户画像管理 ==========
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.storage.get_user_profile(user_id)
    
    def update_user_profile(self, profile: UserProfile):
        """更新用户画像"""
        profile.updated_at = datetime.now()
        self.storage.save_user_profile(profile)
    
    # ========== 记忆维护 ==========
    
    def run_forgetting(self, user_id: str) -> int:
        """运行遗忘机制"""
        return self.storage.delete_weak_episodes(
            user_id, 
            min_strength=self.config.min_importance_threshold
        )
    
    def cleanup(self, days: int = 7) -> int:
        """清理过期数据"""
        return self.storage.cleanup_old_sessions(days)
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取统计信息"""
        return self.storage.get_stats(user_id)
    
    # ========== 导出/导入 ==========
    
    def export_user_memory(self, user_id: str) -> Dict:
        """导出用户所有记忆"""
        profile = self.storage.get_user_profile(user_id)
        episodes = self.storage.get_episodes(user_id, limit=1000)
        facts = self.storage.get_facts(user_id, limit=1000)
        
        return {
            "user_id": user_id,
            "export_time": datetime.now().isoformat(),
            "profile": profile.to_dict() if profile else None,
            "episodes": [ep.to_dict() for ep in episodes],
            "facts": [f.to_dict() for f in facts]
        }
    
    def import_user_memory(self, data: Dict):
        """导入用户记忆"""
        user_id = data["user_id"]
        
        # 导入画像
        if data.get("profile"):
            profile = UserProfile.from_dict(data["profile"])
            self.storage.save_user_profile(profile)
        
        # 导入情景记忆
        for ep_data in data.get("episodes", []):
            episode = Episode.from_dict(ep_data)
            self.storage.save_episode(episode)
        
        # 导入事实
        for fact_data in data.get("facts", []):
            fact = Fact.from_dict(fact_data)
            self.storage.save_fact(fact)
