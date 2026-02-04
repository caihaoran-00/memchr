"""
记忆系统配置文件
针对智能玩具场景优化，平衡效果和成本
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class MemoryConfig:
    """记忆系统配置"""

    # ========== 存储配置 ==========
    # 数据存储路径
    data_dir: str = "./data"
    # SQLite数据库文件名
    db_name: str = "memory.db"

    # ========== 工作记忆配置 ==========
    # 工作记忆最大轮数（滑动窗口）
    working_memory_size: int = 10

    # ========== 情景记忆配置 ==========
    # 单个用户最大情景记忆数量
    max_episodes_per_user: int = 100
    # 情景摘要最大长度（字符）
    episode_summary_max_length: int = 200
    # 触发情景压缩的对话轮数阈值
    episode_compress_threshold: int = 5

    # ========== 语义记忆配置 ==========
    # 用户画像最大标签数
    max_profile_tags: int = 20
    # 最大知识事实数
    max_facts_per_user: int = 50

    # ========== 遗忘机制配置 ==========
    # 记忆衰减周期（天）
    memory_decay_days: int = 30
    # 最低重要性阈值（低于此值将被遗忘）
    min_importance_threshold: float = 0.2
    # 访问次数权重（用于计算记忆强度）
    access_count_weight: float = 0.3
    # 时间衰减权重
    time_decay_weight: float = 0.7

    # ========== LLM配置 ==========
    # LLM API提供商: "openai", "zhipu", "local", "mock"
    llm_provider: str = "openai"
    # API密钥（从环境变量读取）
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    # API基础URL
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )
    # 使用的模型名称
    llm_model: str = "gpt-4o-mini"  # 成本较低的模型
    # 记忆提取使用的模型（可以用更便宜的模型）
    extraction_model: str = "gpt-4o-mini"
    # 最大重试次数
    llm_max_retries: int = 3
    # 请求超时（秒）
    llm_timeout: int = 30

    # ========== 向量检索配置 ==========
    # 是否启用向量检索（可选功能，关闭可降低成本）
    enable_vector_search: bool = False
    # 向量维度
    vector_dim: int = 384  # 使用小模型的维度
    # 本地embedding模型路径
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # 向量相似度阈值
    similarity_threshold: float = 0.7
    # 检索返回的最大结果数
    max_retrieval_results: int = 5

    # ========== 成本控制配置 ==========
    # 批量处理大小（减少API调用次数）
    batch_size: int = 5
    # 缓存过期时间（秒）
    cache_ttl: int = 3600
    # 是否启用本地缓存
    enable_cache: bool = True

    def get_db_path(self) -> str:
        """获取数据库完整路径"""
        return os.path.join(self.data_dir, self.db_name)


# 预设配置模板
class ConfigPresets:
    """配置预设模板"""

    @staticmethod
    def minimal() -> MemoryConfig:
        """最小配置：最低成本，适合测试"""
        return MemoryConfig(
            working_memory_size=5,
            max_episodes_per_user=20,
            max_facts_per_user=10,
            enable_vector_search=False,
            llm_provider="mock",  # 使用mock，不调用真实API
        )

    @staticmethod
    def balanced() -> MemoryConfig:
        """平衡配置：效果和成本的平衡"""
        return MemoryConfig(
            working_memory_size=10,
            max_episodes_per_user=50,
            max_facts_per_user=30,
            enable_vector_search=False,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )

    @staticmethod
    def full_featured() -> MemoryConfig:
        """完整配置：最佳效果，成本较高"""
        return MemoryConfig(
            working_memory_size=15,
            max_episodes_per_user=100,
            max_facts_per_user=50,
            enable_vector_search=True,
            llm_provider="openai",
            llm_model="gpt-4o",
        )


# 默认配置实例
default_config = MemoryConfig()
