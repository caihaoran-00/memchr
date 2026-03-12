"""
记忆系统测试。
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig, PROJECT_CONFIG_EXAMPLE_PATH, PROJECT_CONFIG_PATH
from memory_core.extractor import MemoryExtractor
from memory_core.llm_client import QwenClient, CompositeLLMClient, create_llm_client
from memory_core.manager import MemoryManager
from memory_core.models import (
    Episode,
    Fact,
    MemoryContext,
    Message,
    MessageRole,
    UserProfile,
    WorkingMemory,
)


class DummyLLMClient:
    async def chat(self, messages, temperature=0.7, max_tokens=500):
        return ""

    async def extract_json(self, prompt, schema_hint=""):
        return {}

    async def embed_texts(self, texts):
        return []

    async def rerank(self, query, documents, top_n=None):
        return []


class TestModels:
    def test_message_serialization(self):
        message = Message(role=MessageRole.USER, content="hello")
        restored = Message.from_dict(message.to_dict())
        assert restored.role == MessageRole.USER
        assert restored.content == "hello"

    def test_working_memory_recent_messages(self):
        working_memory = WorkingMemory(user_id="u1", session_id="s1")
        for index in range(3):
            working_memory.add_message(MessageRole.USER, f"user-{index}")
            working_memory.add_message(MessageRole.ASSISTANT, f"assistant-{index}")

        recent = working_memory.get_recent(2)
        assert len(recent) == 4
        assert recent[-1].content == "assistant-2"

    def test_episode_strength(self):
        episode = Episode(user_id="u1", summary="summary", importance=0.8, access_count=4)
        strength = episode.calculate_strength(decay_days=30)
        assert 0.0 <= strength <= 1.0

    def test_profile_add_tag(self):
        profile = UserProfile(user_id="u1")
        profile.add_tag("likes-dinosaurs", max_tags=3)
        profile.add_tag("likes-dinosaurs", max_tags=3)
        assert profile.tags == ["likes-dinosaurs"]

    def test_memory_context_prompt(self):
        profile = UserProfile(user_id="u1", name="xiaoming", age=5, tags=["likes-dinosaurs"])
        facts = [Fact(user_id="u1", subject="xiaoming", predicate="likes", object="dinosaurs")]
        episodes = [Episode(user_id="u1", summary="user talked about dinosaurs")]
        context = MemoryContext(
            user_profile=profile,
            relevant_facts=facts,
            relevant_episodes=episodes,
        )
        prompt = context.to_system_prompt()
        assert "xiaoming" in prompt
        assert "dinosaurs" in prompt


class TestStorage:
    def test_profile_crud(self, storage):
        profile = UserProfile(user_id="user1", name="xiaohong", age=6)
        storage.save_user_profile(profile)

        loaded = storage.get_user_profile("user1")
        assert loaded is not None
        assert loaded.name == "xiaohong"
        assert loaded.age == 6

    def test_episode_embedding_roundtrip(self, storage):
        episode = Episode(
            user_id="user1",
            summary="user likes dinosaurs",
            importance=0.9,
            embedding=[1.0, 0.0, 0.0],
        )
        storage.save_episode(episode)

        loaded = storage.get_episodes("user1", limit=1)[0]
        assert loaded.embedding is not None
        assert loaded.embedding == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_fact_deduplication(self, storage):
        fact = Fact(user_id="user1", subject="xiaoming", predicate="likes", object="dinosaurs")
        storage.save_fact(fact)
        storage.save_fact(fact)
        facts = storage.get_facts("user1")
        assert len(facts) == 1

    def test_keyword_fact_search(self, storage):
        storage.save_fact(Fact(user_id="user1", subject="xiaoming", predicate="likes", object="dinosaurs"))
        storage.save_fact(Fact(user_id="user1", subject="xiaoming", predicate="likes", object="drawing"))
        results = storage.search_facts("user1", "dinosaurs", limit=5)
        assert len(results) == 1
        assert results[0].object == "dinosaurs"


class TestQwenClient:
    @pytest.mark.asyncio
    async def test_extract_json_parsing(self, temp_config, monkeypatch):
        client = QwenClient(
            api_key="test-key",
            base_url=temp_config.llm_base_url,
            model=temp_config.llm_model,
            extraction_model=temp_config.extraction_model,
            embedding_model=temp_config.embedding_model,
            rerank_model=temp_config.rerank_model,
            rerank_api_url=temp_config.rerank_api_url,
            timeout=temp_config.llm_timeout,
            max_retries=temp_config.llm_max_retries,
        )

        async def fake_chat(*args, **kwargs):
            return (
                '```json\n{"summary":"test","keywords":["dinosaurs"],'
                '"emotion":"happy","importance":0.8,"facts":[],'
                '"profile_updates":{"tags":[],"preferences":{}}}\n```'
            )

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await client.extract_json("extract")
        assert result["summary"] == "test"
        assert result["keywords"] == ["dinosaurs"]

    @pytest.mark.asyncio
    async def test_embed_texts_parsing(self, temp_config, monkeypatch):
        client = QwenClient(
            api_key="test-key",
            base_url=temp_config.llm_base_url,
            model=temp_config.llm_model,
            extraction_model=temp_config.extraction_model,
            embedding_model=temp_config.embedding_model,
            rerank_model=temp_config.rerank_model,
            rerank_api_url=temp_config.rerank_api_url,
            timeout=temp_config.llm_timeout,
            max_retries=temp_config.llm_max_retries,
        )

        async def fake_post(url, payload):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

        monkeypatch.setattr(client, "_post_json", fake_post)
        embeddings = await client.embed_texts(["a", "b"])
        assert embeddings == [[1.0, 0.0], [0.0, 1.0]]

    @pytest.mark.asyncio
    async def test_rerank_parsing(self, temp_config, monkeypatch):
        client = QwenClient(
            api_key="test-key",
            base_url=temp_config.llm_base_url,
            model=temp_config.llm_model,
            extraction_model=temp_config.extraction_model,
            embedding_model=temp_config.embedding_model,
            rerank_model=temp_config.rerank_model,
            rerank_api_url=temp_config.rerank_api_url,
            timeout=temp_config.llm_timeout,
            max_retries=temp_config.llm_max_retries,
        )

        async def fake_post(url, payload):
            return {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }
            }

        monkeypatch.setattr(client, "_post_json", fake_post)
        results = await client.rerank("dinosaurs", ["doc-a", "doc-b"], top_n=2)
        assert results[0]["index"] == 1
        assert results[0]["score"] == pytest.approx(0.9)

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError):
            QwenClient(
                api_key="",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen-plus",
                extraction_model="qwen-plus",
                embedding_model="text-embedding-v4",
                rerank_model="qwen3-rerank",
                rerank_api_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                timeout=30,
                max_retries=3,
            )

    def test_provider_factory_can_build_composite_client(self):
        config = MemoryConfig(
            llm_provider="deepseek",
            llm_api_key="deepseek-test",
            embedding_provider="qwen",
            embedding_api_key="qwen-test",
            rerank_provider="qwen",
            rerank_api_key="qwen-test",
        )
        client = create_llm_client(config)
        assert isinstance(client, CompositeLLMClient)


class TestExtractor:
    def test_fact_normalization_deduplicates_semantic_duplicates(self, temp_config):
        extractor = MemoryExtractor(temp_config, llm_client=DummyLLMClient())
        extraction = {
            "facts": [
                {"subject": "小明", "predicate": "年龄", "object": "5岁", "confidence": 0.8},
                {"subject": "小明", "predicate": "年龄是", "object": "5岁", "confidence": 0.9},
                {"subject": "小明", "predicate": "名字", "object": "是小明", "confidence": 0.7},
            ]
        }

        facts = extractor.create_facts_from_extraction(extraction, "user1", "session1")

        assert len(facts) == 2
        assert any(fact.predicate == "年龄是" and fact.object == "5岁" for fact in facts)
        assert any(fact.predicate == "名字是" and fact.object == "小明" for fact in facts)


class TestConfig:
    def test_conversation_turn_alias(self):
        config = MemoryConfig(conversation_turns_before_extraction=3)
        assert config.episode_compress_threshold == 3
        assert config.conversation_turns_before_extraction == 3

    def test_project_settings_file_exists(self):
        assert os.path.exists(PROJECT_CONFIG_EXAMPLE_PATH)

    def test_project_settings_json_is_valid(self):
        with open(PROJECT_CONFIG_EXAMPLE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        assert "llm_provider" in data
        assert "conversation_turns_before_extraction" in data
        assert "provider_api_keys" in data

        # Optional: if a local private settings file exists, it should also be valid JSON.
        if os.path.exists(PROJECT_CONFIG_PATH):
            with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as file:
                json.load(file)


class TestMemoryManager:
    def test_session_lifecycle(self, manager):
        working_memory = manager.start_session("user1")
        manager.add_message(working_memory.session_id, "user", "hello")
        manager.add_message(working_memory.session_id, "assistant", "hi")
        assert len(manager._working_memory_cache[working_memory.session_id].messages) == 2

    @pytest.mark.asyncio
    async def test_qwen_extraction_generates_episode_fact_and_profile(self, manager):
        session = manager.start_session("child_a")
        manager.add_message(session.session_id, "user", "我叫小明，我5岁了，我最喜欢恐龙。")
        manager.add_message(session.session_id, "assistant", "太好了，你最喜欢哪种恐龙？")
        manager.add_message(session.session_id, "user", "我喜欢霸王龙。")
        manager.add_message(session.session_id, "assistant", "霸王龙真的很酷。")

        episode = await manager.end_session(session.session_id, extract_memory=True)
        assert episode is not None
        assert episode.embedding is not None
        assert "恐龙" in episode.summary

        profile = manager.get_user_profile("child_a")
        assert profile is not None
        assert profile.name == "小明"
        assert "喜欢恐龙" in profile.tags

        facts = manager.storage.get_facts("child_a")
        assert len(facts) >= 1
        assert any(fact.predicate == "喜欢" and fact.object == "恐龙" for fact in facts)

    @pytest.mark.asyncio
    async def test_episode_semantic_retrieval_with_rerank(self, manager):
        first = Episode(
            user_id="user_semantic",
            summary="用户说自己最喜欢恐龙，尤其喜欢霸王龙。",
            importance=0.7,
            embedding=[1.0, 0.0, 0.0],
            created_at=datetime.now() - timedelta(hours=2),
        )
        second = Episode(
            user_id="user_semantic",
            summary="用户说自己最近常常画画。",
            importance=0.9,
            embedding=[0.0, 1.0, 0.0],
            created_at=datetime.now() - timedelta(days=1),
        )
        manager.storage.save_episode(first)
        manager.storage.save_episode(second)

        session = manager.start_session("user_semantic")
        manager.add_message(session.session_id, "user", "你还记得我最喜欢什么恐龙吗？")
        context = await manager.get_memory_context(session.session_id, "我最喜欢什么恐龙")

        assert context.relevant_episodes
        assert context.relevant_episodes[0].summary == first.summary

    @pytest.mark.asyncio
    async def test_fact_retrieval_keeps_keyword_strategy(self, manager):
        manager.storage.save_fact(Fact(user_id="user_fact", subject="小明", predicate="喜欢", object="恐龙"))
        manager.storage.save_fact(Fact(user_id="user_fact", subject="小明", predicate="喜欢", object="积木"))
        session = manager.start_session("user_fact")
        manager.add_message(session.session_id, "user", "你记得我喜欢什么吗")
        context = await manager.get_memory_context(session.session_id, "恐龙")
        assert len(context.relevant_facts) == 1
        assert context.relevant_facts[0].object == "恐龙"

    @pytest.mark.asyncio
    async def test_fact_retrieval_can_expand_natural_language_query(self, manager):
        manager.storage.save_fact(
            Fact(user_id="user_fact_nl", subject="小明", predicate="喜欢", object="恐龙故事")
        )
        session = manager.start_session("user_fact_nl")
        manager.add_message(session.session_id, "user", "你还记得我最喜欢什么吗")
        context = await manager.get_memory_context(session.session_id, "你还记得我最喜欢什么吗")
        assert any(fact.object == "恐龙故事" for fact in context.relevant_facts)

    def test_export_import_keeps_target_user(self, manager):
        source_user = "export_user"
        target_user = "import_user"

        manager.update_user_profile(UserProfile(user_id=source_user, name="小明"))
        manager.storage.save_episode(Episode(user_id=source_user, summary="source episode"))
        manager.storage.save_fact(Fact(user_id=source_user, subject="小明", predicate="喜欢", object="恐龙"))

        data = manager.export_user_memory(source_user)
        data["user_id"] = target_user
        data["profile"]["user_id"] = target_user
        manager.import_user_memory(data)

        imported_profile = manager.get_user_profile(target_user)
        imported_episodes = manager.storage.get_episodes(target_user)
        imported_facts = manager.storage.get_facts(target_user)

        assert imported_profile is not None
        assert imported_profile.user_id == target_user
        assert imported_episodes[0].user_id == target_user
        assert imported_facts[0].user_id == target_user

    @pytest.mark.asyncio
    async def test_qwen_errors_are_not_silently_downgraded(self, temp_config, monkeypatch):
        class BrokenQwenClient:
            async def extract_json(self, prompt, schema_hint=""):
                raise RuntimeError("qwen failed")

            async def embed_texts(self, texts):
                return []

            async def rerank(self, query, documents, top_n=None):
                return []

            async def chat(self, messages, temperature=0.7, max_tokens=500, model=None):
                return ""

        monkeypatch.setattr(
            "memory_core.manager.create_llm_client",
            lambda config: BrokenQwenClient(),
        )
        broken_manager = MemoryManager(temp_config)
        session = broken_manager.start_session("broken_user")
        for _ in range(2):
            broken_manager.add_message(session.session_id, "user", "我叫小明，我喜欢恐龙。")
            broken_manager.add_message(session.session_id, "assistant", "好的，我记住了。")

        with pytest.raises(RuntimeError, match="qwen failed"):
            await broken_manager.end_session(session.session_id, extract_memory=True)
