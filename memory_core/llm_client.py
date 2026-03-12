"""
LLM 客户端工厂。
"""

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig


class LLMClient(ABC):
    """LLM 客户端抽象。"""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """聊天。"""

    @abstractmethod
    async def extract_json(self, prompt: str, schema_hint: str = "") -> Dict[str, Any]:
        """抽取 JSON。"""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """文本向量化。"""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """重排。"""


class OpenAICompatibleClient(LLMClient):
    """OpenAI 兼容接口客户端。"""

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        extraction_model: Optional[str],
        timeout: int,
        max_retries: int,
    ):
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.extraction_model = extraction_model or model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(f"{provider_name} API key 未配置")
        if not self.base_url:
            raise ValueError(f"{provider_name} base_url 未配置")
        if not self.model:
            raise ValueError(f"{provider_name} model 未配置")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: Optional[str] = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._post_json(f"{self.base_url}/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    async def extract_json(self, prompt: str, schema_hint: str = "") -> Dict[str, Any]:
        system_prompt = (
            "你是一个负责结构化记忆抽取的助手。"
            "请严格返回 JSON，不要输出额外解释。\n"
            f"{schema_hint}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self.chat(
            messages,
            temperature=0.1,
            max_tokens=800,
            model=self.extraction_model,
        )

        try:
            if "```json" in response:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response:
                response = response.split("```", 1)[1].split("```", 1)[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError(f"{self.provider_name} 未配置 embedding 能力")

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        raise RuntimeError(f"{self.provider_name} 未配置 rerank 能力")

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(attempt + 1)

        raise RuntimeError(f"{self.provider_name} 请求失败: {last_error}")


class QwenClient(OpenAICompatibleClient):
    """Qwen 客户端。"""

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        extraction_model: Optional[str],
        embedding_model: Optional[str],
        rerank_model: Optional[str],
        rerank_api_url: Optional[str],
        timeout: int,
        max_retries: int,
    ):
        super().__init__(
            provider_name="Qwen",
            api_key=api_key,
            base_url=base_url,
            model=model,
            extraction_model=extraction_model,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.rerank_api_url = rerank_api_url

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self.embedding_model, "input": texts}
        data = await self._post_json(f"{self.base_url}/embeddings", payload)
        embeddings = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [item.get("embedding", []) for item in embeddings]

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        payload = {
            "model": self.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n or len(documents)},
        }
        data = await self._post_json(self.rerank_api_url, payload)
        results = data.get("output", {}).get("results", [])

        normalized = []
        for result in results:
            index = result.get("index", 0)
            normalized.append(
                {
                    "index": index,
                    "score": float(result.get("relevance_score", 0.0)),
                    "document": documents[index] if 0 <= index < len(documents) else "",
                }
            )
        return normalized


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek 客户端，仅负责聊天和记忆提取。"""


class CompositeLLMClient(LLMClient):
    """组合客户端，用不同 provider 承担不同能力。"""

    def __init__(
        self,
        chat_client: LLMClient,
        embedding_client: Optional[LLMClient] = None,
        rerank_client: Optional[LLMClient] = None,
    ):
        self.chat_client = chat_client
        self.embedding_client = embedding_client or chat_client
        self.rerank_client = rerank_client or self.embedding_client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        return await self.chat_client.chat(messages, temperature, max_tokens)

    async def extract_json(self, prompt: str, schema_hint: str = "") -> Dict[str, Any]:
        return await self.chat_client.extract_json(prompt, schema_hint)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self.embedding_client.embed_texts(texts)

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return await self.rerank_client.rerank(query, documents, top_n)


def _build_qwen_client(
    config: MemoryConfig,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    rerank_api_url: Optional[str],
) -> QwenClient:
    return QwenClient(
        api_key=api_key,
        base_url=base_url,
        model=config.llm_model,
        extraction_model=config.extraction_model,
        embedding_model=config.embedding_model,
        rerank_model=config.rerank_model,
        rerank_api_url=rerank_api_url,
        timeout=config.llm_timeout,
        max_retries=config.llm_max_retries,
    )


def _create_chat_client(config: MemoryConfig) -> LLMClient:
    if config.llm_provider == "qwen":
        return _build_qwen_client(
            config,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            rerank_api_url=config.rerank_api_url,
        )

    if config.llm_provider == "deepseek":
        return DeepSeekClient(
            provider_name="DeepSeek",
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            extraction_model=config.extraction_model,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
        )

    raise ValueError(f"不支持的 llm_provider: {config.llm_provider}")


def _create_embedding_client(config: MemoryConfig) -> Optional[LLMClient]:
    if not config.enable_vector_search:
        return None
    if config.embedding_provider == "qwen":
        return _build_qwen_client(
            config,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            rerank_api_url=config.rerank_api_url,
        )

    raise ValueError(f"不支持的 embedding_provider: {config.embedding_provider}")


def _create_rerank_client(config: MemoryConfig) -> Optional[LLMClient]:
    if not config.enable_rerank:
        return None
    if config.rerank_provider == "qwen":
        return _build_qwen_client(
            config,
            api_key=config.rerank_api_key,
            base_url=config.embedding_base_url,
            rerank_api_url=config.rerank_api_url,
        )

    raise ValueError(f"不支持的 rerank_provider: {config.rerank_provider}")


def create_llm_client(config: MemoryConfig) -> LLMClient:
    """根据配置创建 LLM 客户端。"""
    chat_client = _create_chat_client(config)
    embedding_client = _create_embedding_client(config)
    rerank_client = _create_rerank_client(config)

    if (
        embedding_client is None
        and rerank_client is None
        and isinstance(chat_client, (QwenClient, DeepSeekClient))
    ):
        return chat_client

    return CompositeLLMClient(chat_client, embedding_client, rerank_client)
