"""
pytest 公共 fixture。
"""

import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigPresets
from memory_core.manager import MemoryManager
from storage.sqlite_storage import SQLiteStorage


class FakeQwenClient:
    """测试用 Qwen 替身，覆盖提取、embedding 和 rerank。"""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: Optional[str] = None,
    ) -> str:
        return "测试回复"

    async def extract_json(self, prompt: str, schema_hint: str = "") -> Dict[str, Any]:
        if "小明" in prompt and "恐龙" in prompt:
            return {
                "summary": "用户介绍自己叫小明，并表达了对恐龙的喜爱。",
                "keywords": ["小明", "恐龙", "喜欢"],
                "emotion": "开心",
                "importance": 0.88,
                "facts": [
                    {
                        "subject": "小明",
                        "predicate": "喜欢",
                        "object": "恐龙",
                        "confidence": 0.96,
                    }
                ],
                "profile_updates": {
                    "name": "小明",
                    "age": 5,
                    "gender": "",
                    "tags": ["喜欢恐龙", "爱聊天"],
                    "preferences": {"favorite_topic": "恐龙"},
                },
            }

        if "画画" in prompt:
            return {
                "summary": "用户提到自己喜欢画画，尤其喜欢画小动物。",
                "keywords": ["画画", "小动物"],
                "emotion": "平静",
                "importance": 0.72,
                "facts": [
                    {
                        "subject": "用户",
                        "predicate": "喜欢",
                        "object": "画画",
                        "confidence": 0.9,
                    }
                ],
                "profile_updates": {
                    "name": "",
                    "age": None,
                    "gender": "",
                    "tags": ["喜欢画画"],
                    "preferences": {"hobby": "画画"},
                },
            }

        return {
            "summary": "用户完成了一段普通对话。",
            "keywords": ["对话"],
            "emotion": "平静",
            "importance": 0.5,
            "facts": [],
            "profile_updates": {"tags": [], "preferences": {}},
        }

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            vector = [0.0, 0.0, 0.0]
            if "恐龙" in text:
                vector[0] = 1.0
            if "画画" in text or "动物" in text:
                vector[1] = 1.0
            if vector == [0.0, 0.0, 0.0]:
                vector[2] = 1.0
            embeddings.append(vector)
        return embeddings

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        scored = []
        for index, document in enumerate(documents):
            score = 0.1
            if "恐龙" in query and "恐龙" in document:
                score = 0.95
            elif "画画" in query and "画画" in document:
                score = 0.9
            scored.append({"index": index, "score": score, "document": document})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: top_n or len(scored)]


@pytest.fixture
def temp_config():
    """创建临时配置和数据目录。"""
    temp_dir = tempfile.mkdtemp()
    config = ConfigPresets.full_featured()
    config.data_dir = temp_dir
    config.llm_api_key = "test-qwen-key"
    config.episode_compress_threshold = 2
    yield config
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def storage(temp_config):
    return SQLiteStorage(temp_config)


@pytest.fixture
def fake_qwen_client():
    return FakeQwenClient()


@pytest.fixture
def manager(temp_config, fake_qwen_client, monkeypatch):
    monkeypatch.setattr("memory_core.manager.create_llm_client", lambda config: fake_qwen_client)
    return MemoryManager(temp_config)


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
