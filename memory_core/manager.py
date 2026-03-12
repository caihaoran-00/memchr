"""
记忆系统核心管理器。
"""

import math
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig
from memory_core.extractor import MemoryExtractor, create_extractor
from memory_core.llm_client import LLMClient, create_llm_client
from memory_core.models import (
    Episode,
    Fact,
    MemoryContext,
    MessageRole,
    UserProfile,
    WorkingMemory,
)
from storage.sqlite_storage import SQLiteStorage


FACT_QUERY_STOP_TERMS = {
    "我",
    "你",
    "他",
    "她",
    "记得",
    "知道",
    "还有",
    "一下",
    "之前",
    "现在",
}

FACT_SUMMARY_HINTS = [
    "我喜欢什么",
    "我最喜欢什么",
    "你还记得我喜欢什么",
    "我说过什么",
    "我有哪些喜好",
    "我有什么偏好",
]


class MemoryManager:
    """负责会话管理、记忆提取、检索和导入导出。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.storage = SQLiteStorage(self.config)
        self.llm_client: LLMClient = create_llm_client(self.config)
        self.extractor: MemoryExtractor = create_extractor(
            self.config, llm_client=self.llm_client
        )
        self._working_memory_cache: Dict[str, WorkingMemory] = {}

    # ========== 会话 ==========

    def start_session(
        self, user_id: str, session_id: Optional[str] = None
    ) -> WorkingMemory:
        """开始或恢复一个会话。"""
        session_id = session_id or str(uuid.uuid4())

        existing = self.storage.get_working_memory(session_id)
        if existing:
            self._working_memory_cache[session_id] = existing
            return existing

        working_memory = WorkingMemory(user_id=user_id, session_id=session_id)
        self._working_memory_cache[session_id] = working_memory
        self.storage.save_working_memory(working_memory)
        return working_memory

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向工作记忆追加一条消息。"""
        working_memory = self._get_working_memory(session_id)
        msg_role = MessageRole(role) if isinstance(role, str) else role
        working_memory.add_message(msg_role, content, metadata or {})

        max_messages = self.config.working_memory_size * 2
        if len(working_memory.messages) > max_messages:
            working_memory.messages = working_memory.messages[-max_messages:]

        self.storage.save_working_memory(working_memory)

    async def end_session(
        self, session_id: str, extract_memory: bool = True
    ) -> Optional[Episode]:
        """结束会话，并在达到阈值时提取长期记忆。"""
        working_memory = self._working_memory_cache.get(session_id)
        if not working_memory:
            working_memory = self.storage.get_working_memory(session_id)
        if not working_memory:
            return None

        episode = None
        if (
            extract_memory
            and len(working_memory.messages)
            >= self.config.episode_compress_threshold * 2
        ):
            episode = await self._extract_and_store_memory(working_memory)

        self._working_memory_cache.pop(session_id, None)
        self.storage.delete_working_memory(session_id)
        return episode

    async def _extract_and_store_memory(self, working_memory: WorkingMemory) -> Episode:
        """从工作记忆中提取 Episode、Fact 和 UserProfile。"""
        user_id = working_memory.user_id
        session_id = working_memory.session_id

        extraction = await self.extractor.extract_from_conversation(
            working_memory.messages, user_id, session_id
        )

        episode = await self.extractor.create_episode_from_extraction(
            extraction, user_id, session_id
        )
        self.storage.save_episode(episode)

        for fact in self.extractor.create_facts_from_extraction(
            extraction, user_id, session_id
        ):
            self.storage.save_fact(fact)

        profile = self.storage.get_user_profile(user_id) or UserProfile(user_id=user_id)
        profile = self.extractor.update_profile_from_extraction(profile, extraction)
        self.storage.save_user_profile(profile)

        return episode

    # ========== 检索 ==========

    async def get_memory_context(
        self, session_id: str, query: Optional[str] = None
    ) -> MemoryContext:
        """获取当前会话需要注入给上层模型的记忆上下文。"""
        working_memory = self._working_memory_cache.get(session_id)
        if not working_memory:
            working_memory = self.storage.get_working_memory(session_id)
        if not working_memory:
            return MemoryContext()

        user_id = working_memory.user_id
        profile = self.storage.get_user_profile(user_id)
        episodes = await self._retrieve_relevant_episodes(user_id, query)
        facts = self._retrieve_relevant_facts(user_id, query)

        for episode in episodes:
            self.storage.update_episode_access(episode.id)

        return MemoryContext(
            working_memory=working_memory,
            relevant_episodes=episodes,
            user_profile=profile,
            relevant_facts=facts,
        )

    async def _retrieve_relevant_episodes(
        self, user_id: str, query: Optional[str]
    ) -> List[Episode]:
        if not query:
            return self.storage.get_episodes(
                user_id,
                limit=self.config.episode_rerank_top_n,
                min_importance=self.config.min_importance_threshold,
            )

        if not self.config.enable_vector_search:
            return self.storage.search_episodes_by_keywords(
                user_id, [query], limit=self.config.episode_rerank_top_n
            )

        candidates = self.storage.get_episode_candidates(
            user_id, limit=max(self.config.episode_retrieval_top_k * 3, 20)
        )
        if not candidates:
            return []

        await self._ensure_episode_embeddings(candidates)

        query_embeddings = await self.llm_client.embed_texts([query])
        if not query_embeddings:
            return []
        query_embedding = query_embeddings[0]

        scored = []
        for episode in candidates:
            if not episode.embedding:
                continue
            semantic_score = self._cosine_similarity(query_embedding, episode.embedding)
            if semantic_score < self.config.similarity_threshold:
                continue
            scored.append((episode, semantic_score))

        if not scored:
            return []

        scored.sort(key=lambda item: item[1], reverse=True)
        top_scored = scored[: self.config.episode_retrieval_top_k]

        rerank_scores: Dict[str, float] = {}
        if self.config.enable_rerank:
            rerank_scores = await self._rerank_episodes(query, top_scored)

        final_ranked = []
        for episode, semantic_score in top_scored:
            final_score = self._calculate_episode_score(
                episode=episode,
                semantic_score=semantic_score,
                rerank_score=rerank_scores.get(episode.id, 0.0),
            )
            final_ranked.append((episode, final_score))

        final_ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            episode
            for episode, _score in final_ranked[: self.config.episode_rerank_top_n]
        ]

    def _retrieve_relevant_facts(self, user_id: str, query: Optional[str]) -> List[Fact]:
        if query:
            results = self.storage.search_facts(
                user_id, query, limit=self.config.max_context_facts
            )
            if results:
                return results

            expanded_terms = self._expand_fact_queries(query)
            merged = self._search_facts_by_terms(user_id, expanded_terms)
            if merged:
                return merged

            if (not expanded_terms) or self._is_fact_summary_query(query):
                return self.storage.get_facts(
                    user_id, limit=self.config.max_context_facts
                )
            return []
        return self.storage.get_facts(user_id, limit=self.config.max_context_facts)

    def _search_facts_by_terms(self, user_id: str, terms: Sequence[str]) -> List[Fact]:
        merged: Dict[str, Fact] = {}
        for term in terms:
            for fact in self.storage.search_facts(
                user_id, term, limit=self.config.max_context_facts
            ):
                merged[fact.id] = fact
            if len(merged) >= self.config.max_context_facts:
                break
        return list(merged.values())[: self.config.max_context_facts]

    def _expand_fact_queries(self, query: str) -> List[str]:
        terms = self._split_query_terms(query)
        expanded: List[str] = []
        for term in terms:
            if term in FACT_QUERY_STOP_TERMS:
                continue
            if len(term) >= 2:
                expanded.append(term)
            trimmed = term
            for stop in FACT_QUERY_STOP_TERMS:
                trimmed = trimmed.replace(stop, "")
            if len(trimmed) >= 2:
                expanded.append(trimmed)
        return self._dedupe_terms(expanded)

    def _split_query_terms(self, query: str) -> List[str]:
        terms: List[str] = []
        buffer = ""
        for char in query:
            if "\u4e00" <= char <= "\u9fff" or char.isalnum():
                buffer += char
                continue
            if buffer:
                terms.append(buffer)
                buffer = ""
        if buffer:
            terms.append(buffer)
        return terms

    def _dedupe_terms(self, terms: Sequence[str]) -> List[str]:
        seen = set()
        ordered = []
        for term in terms:
            if not term or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered

    def _is_fact_summary_query(self, query: str) -> bool:
        return any(hint in query for hint in FACT_SUMMARY_HINTS)

    async def _ensure_episode_embeddings(self, episodes: Sequence[Episode]) -> None:
        missing = [episode for episode in episodes if episode.summary and not episode.embedding]
        if not missing:
            return

        embeddings = await self.llm_client.embed_texts(
            [episode.summary for episode in missing]
        )
        for episode, embedding in zip(missing, embeddings):
            episode.embedding = embedding
            self.storage.save_episode(episode)

    async def _rerank_episodes(
        self, query: str, scored_episodes: Sequence[Tuple[Episode, float]]
    ) -> Dict[str, float]:
        documents = [episode.summary for episode, _semantic in scored_episodes]
        results = await self.llm_client.rerank(
            query, documents, top_n=min(self.config.episode_rerank_top_n, len(documents))
        )

        rerank_scores: Dict[str, float] = {}
        for result in results:
            index = result.get("index", 0)
            if 0 <= index < len(scored_episodes):
                rerank_scores[scored_episodes[index][0].id] = float(
                    result.get("score", 0.0)
                )
        return rerank_scores

    def _calculate_episode_score(
        self, episode: Episode, semantic_score: float, rerank_score: float
    ) -> float:
        importance_score = max(0.0, min(1.0, episode.importance))
        recency_score = self._recency_score(episode)

        if self.config.enable_rerank:
            return (
                semantic_score * 0.45
                + rerank_score * 0.30
                + importance_score * 0.15
                + recency_score * 0.10
            )

        return semantic_score * 0.70 + importance_score * 0.20 + recency_score * 0.10

    def _recency_score(self, episode: Episode) -> float:
        age_seconds = max((datetime.now() - episode.created_at).total_seconds(), 0.0)
        one_week = 7 * 24 * 60 * 60
        return max(0.0, 1.0 - age_seconds / one_week)

    def _cosine_similarity(
        self, left: Sequence[float], right: Sequence[float]
    ) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _get_working_memory(self, session_id: str) -> WorkingMemory:
        working_memory = self._working_memory_cache.get(session_id)
        if working_memory:
            return working_memory

        working_memory = self.storage.get_working_memory(session_id)
        if not working_memory:
            raise ValueError(f"会话不存在: {session_id}")

        self._working_memory_cache[session_id] = working_memory
        return working_memory

    # ========== 用户画像 ==========

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        return self.storage.get_user_profile(user_id)

    def update_user_profile(self, profile: UserProfile) -> None:
        profile.updated_at = datetime.now()
        self.storage.save_user_profile(profile)

    # ========== 维护 ==========

    def run_forgetting(self, user_id: str) -> int:
        return self.storage.delete_weak_episodes(
            user_id, min_strength=self.config.min_importance_threshold
        )

    def cleanup(self, days: int = 7) -> int:
        return self.storage.cleanup_old_sessions(days)

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        return self.storage.get_stats(user_id)

    # ========== 导出/导入 ==========

    def export_user_memory(self, user_id: str) -> Dict[str, Any]:
        profile = self.storage.get_user_profile(user_id)
        episodes = self.storage.get_episodes(user_id, limit=1000)
        facts = self.storage.get_facts(user_id, limit=1000)

        return {
            "user_id": user_id,
            "export_time": datetime.now().isoformat(),
            "profile": profile.to_dict() if profile else None,
            "episodes": [episode.to_dict() for episode in episodes],
            "facts": [fact.to_dict() for fact in facts],
        }

    def import_user_memory(self, data: Dict[str, Any]) -> None:
        user_id = data["user_id"]

        if data.get("profile"):
            profile = UserProfile.from_dict(data["profile"])
            profile.user_id = user_id
            profile.updated_at = datetime.now()
            self.storage.save_user_profile(profile)

        for episode_data in data.get("episodes", []):
            episode = Episode.from_dict(episode_data)
            if episode.user_id != user_id:
                episode.id = str(uuid.uuid4())
            episode.user_id = user_id
            self.storage.save_episode(episode)

        for fact_data in data.get("facts", []):
            fact = Fact.from_dict(fact_data)
            if fact.user_id != user_id:
                fact.id = str(uuid.uuid4())
            fact.user_id = user_id
            self.storage.save_fact(fact)
