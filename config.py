"""
记忆系统配置管理。
"""

from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Optional
import json
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "memory_settings.json")
PROJECT_ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def _load_project_env() -> None:
    """从项目根目录加载 .env，不覆盖已有环境变量。"""
    if not os.path.exists(PROJECT_ENV_PATH):
        return

    with open(PROJECT_ENV_PATH, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_project_settings() -> Dict[str, Any]:
    """从项目根目录加载统一配置文件。"""
    if not os.path.exists(PROJECT_CONFIG_PATH):
        return {}

    with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


_load_project_env()
PROJECT_SETTINGS = _load_project_settings()


PROVIDER_DEFAULTS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "embedding_model": "text-embedding-v4",
        "rerank_model": "qwen3-rerank",
        "rerank_api_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "embedding_model": "text-embedding-v4",
        "rerank_model": "qwen3-rerank",
        "rerank_api_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    },
}


def _setting(name: str, default: Any = None) -> Any:
    return PROJECT_SETTINGS.get(name, default)


def _provider_key(provider: str, generic_env: str, explicit_value: Optional[str]) -> Optional[str]:
    if explicit_value:
        return explicit_value

    provider = provider.lower()
    provider_keys = _setting("provider_api_keys", {})
    if isinstance(provider_keys, dict):
        key = provider_keys.get(provider)
        if key:
            return key

    if provider == "qwen":
        return (
            os.getenv("QWEN_API_KEY")
            or os.getenv(generic_env)
            or os.getenv("LLM_API_KEY")
        )
    if provider == "deepseek":
        return (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv(generic_env)
            or os.getenv("LLM_API_KEY")
        )
    return os.getenv(generic_env) or os.getenv("LLM_API_KEY")


def _provider_default(provider: str, field_name: str) -> Optional[str]:
    return PROVIDER_DEFAULTS.get(provider.lower(), {}).get(field_name)


def _field_default(field_name: str) -> Any:
    for field in fields(MemoryConfig):
        if field.name != field_name:
            continue
        if field.default is not MISSING:
            return field.default
        if field.default_factory is not MISSING:  # type: ignore[attr-defined]
            return field.default_factory()
    raise KeyError(field_name)


@dataclass
class MemoryConfig:
    """记忆系统配置。"""

    data_dir: str = "./data"
    db_name: str = "memory.db"

    working_memory_size: int = 10

    max_episodes_per_user: int = 100
    episode_summary_max_length: int = 200
    episode_compress_threshold: int = 5
    conversation_turns_before_extraction: Optional[int] = None
    max_context_episodes: int = 5

    max_profile_tags: int = 20
    max_facts_per_user: int = 50
    max_context_facts: int = 5

    memory_decay_days: int = 30
    min_importance_threshold: float = 0.2
    access_count_weight: float = 0.3
    time_decay_weight: float = 0.7

    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    extraction_model: Optional[str] = None
    llm_max_retries: int = 3
    llm_timeout: int = 30

    enable_vector_search: bool = True
    vector_dim: int = 1024
    embedding_provider: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_model: Optional[str] = None

    enable_rerank: bool = True
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_model: Optional[str] = None
    rerank_api_url: Optional[str] = None

    similarity_threshold: float = 0.45
    max_retrieval_results: int = 5
    episode_retrieval_top_k: int = 10
    episode_rerank_top_n: int = 5

    batch_size: int = 5
    cache_ttl: int = 3600
    enable_cache: bool = True

    def __post_init__(self) -> None:
        self._apply_project_overrides(
            "data_dir",
            "db_name",
            "working_memory_size",
            "max_episodes_per_user",
            "episode_summary_max_length",
            "episode_compress_threshold",
            "max_context_episodes",
            "max_profile_tags",
            "max_facts_per_user",
            "max_context_facts",
            "memory_decay_days",
            "min_importance_threshold",
            "access_count_weight",
            "time_decay_weight",
            "llm_max_retries",
            "llm_timeout",
            "vector_dim",
            "similarity_threshold",
            "max_retrieval_results",
            "episode_retrieval_top_k",
            "episode_rerank_top_n",
            "batch_size",
            "cache_ttl",
        )
        self._apply_project_overrides(
            "enable_vector_search",
            "enable_rerank",
            "enable_cache",
            default_value=True,
        )
        self.conversation_turns_before_extraction = (
            self.conversation_turns_before_extraction
            or _setting("conversation_turns_before_extraction")
        )
        if self.conversation_turns_before_extraction is not None:
            self.episode_compress_threshold = self.conversation_turns_before_extraction

        self.llm_provider = (
            self.llm_provider
            or _setting("llm_provider")
            or os.getenv("LLM_PROVIDER")
            or "qwen"
        ).lower()
        self.embedding_provider = (
            self.embedding_provider
            or _setting("embedding_provider")
            or os.getenv("EMBEDDING_PROVIDER")
            or "qwen"
        ).lower()
        self.rerank_provider = (
            self.rerank_provider
            or _setting("rerank_provider")
            or os.getenv("RERANK_PROVIDER")
            or "qwen"
        ).lower()

        self.llm_api_key = _provider_key(self.llm_provider, "LLM_API_KEY", self.llm_api_key)
        self.embedding_api_key = _provider_key(
            self.embedding_provider, "EMBEDDING_API_KEY", self.embedding_api_key
        )
        self.rerank_api_key = _provider_key(
            self.rerank_provider, "RERANK_API_KEY", self.rerank_api_key
        )

        self.llm_base_url = (
            self.llm_base_url
            or _setting("llm_base_url")
            or os.getenv("LLM_BASE_URL")
            or _provider_default(self.llm_provider, "base_url")
        )
        self.embedding_base_url = (
            self.embedding_base_url
            or _setting("embedding_base_url")
            or os.getenv("EMBEDDING_BASE_URL")
            or _provider_default(self.embedding_provider, "base_url")
        )
        self.rerank_api_url = (
            self.rerank_api_url
            or _setting("rerank_api_url")
            or os.getenv("RERANK_API_URL")
            or _provider_default(self.rerank_provider, "rerank_api_url")
        )

        self.llm_model = (
            self.llm_model
            or _setting("llm_model")
            or os.getenv("LLM_MODEL")
            or _provider_default(self.llm_provider, "model")
        )
        self.extraction_model = (
            self.extraction_model
            or _setting("extraction_model")
            or os.getenv("EXTRACTION_MODEL")
            or self.llm_model
        )
        self.embedding_model = (
            self.embedding_model
            or _setting("embedding_model")
            or os.getenv("EMBEDDING_MODEL")
            or _provider_default(self.embedding_provider, "embedding_model")
        )
        self.rerank_model = (
            self.rerank_model
            or _setting("rerank_model")
            or os.getenv("RERANK_MODEL")
            or _provider_default(self.rerank_provider, "rerank_model")
        )

        self.conversation_turns_before_extraction = self.episode_compress_threshold

    def _apply_project_overrides(
        self, *field_names: str, default_value: Any = None
    ) -> None:
        for field_name in field_names:
            field_default = (
                _field_default(field_name) if default_value is None else default_value
            )
            current_value = getattr(self, field_name)
            if current_value == field_default:
                setattr(self, field_name, _setting(field_name, field_default))

    def get_db_path(self) -> str:
        return os.path.join(self.data_dir, self.db_name)


class ConfigPresets:
    """预设配置。"""

    @staticmethod
    def minimal() -> MemoryConfig:
        return MemoryConfig(
            working_memory_size=5,
            max_episodes_per_user=20,
            max_facts_per_user=10,
            enable_vector_search=False,
            enable_rerank=False,
        )

    @staticmethod
    def balanced() -> MemoryConfig:
        return MemoryConfig(
            working_memory_size=10,
            max_episodes_per_user=50,
            max_facts_per_user=30,
            enable_vector_search=True,
            enable_rerank=True,
        )

    @staticmethod
    def full_featured() -> MemoryConfig:
        return MemoryConfig(
            working_memory_size=15,
            max_episodes_per_user=100,
            max_facts_per_user=50,
            enable_vector_search=True,
            enable_rerank=True,
            episode_retrieval_top_k=20,
            episode_rerank_top_n=8,
        )


default_config = MemoryConfig()
