# AGENTS.md - 项目知识库

**Generated:** 2026-03-12
**Project:** toy-memory-system

## OVERVIEW

儿童智能对话玩具记忆系统，当前使用 Qwen 完整接管记忆提取与 `Episode` 语义检索。

三层记忆：

- `WorkingMemory`：当前对话窗口
- `Episode`：情景记忆摘要
- `Fact` / `UserProfile`：长期事实与画像

当前实现要点：

- 记忆提取只走 Qwen
- `Episode` 检索走 `embedding + rerank`
- `Fact` 检索走关键词/结构化匹配
- 不再保留 `RuleBasedExtractor` 和 `mock provider`

## STRUCTURE

```text
.
├── config.py
├── memory_core/
│   ├── models.py
│   ├── manager.py
│   ├── extractor.py
│   ├── llm_client.py
│   └── AGENTS.md
├── storage/
│   └── sqlite_storage.py
├── api/
│   └── server.py
├── tests/
│   ├── conftest.py
│   ├── test_memory.py
│   └── demo.py
└── examples/
    └── integration_example.py
```

## WHERE TO LOOK

| 任务 | 位置 | 说明 |
|------|------|------|
| 改记忆模型 | `memory_core/models.py` | `@dataclass` + `to_dict/from_dict` |
| 改提取流程 | `memory_core/extractor.py` | Qwen JSON 抽取 |
| 改检索流程 | `memory_core/manager.py` | `Episode` 语义召回 + `Fact` 关键词召回 |
| 改 Qwen 接入 | `memory_core/llm_client.py` | `chat/extract_json/embed_texts/rerank` |
| 改 SQLite | `storage/sqlite_storage.py` | `episodes.embedding` 持久化 |
| 改 API | `api/server.py` | FastAPI 路由 |
| 改测试 | `tests/test_memory.py` | 单测和集成测试 |

## CODE MAP

| 符号 | 类型 | 文件 | 作用 |
|------|------|------|------|
| `MemoryManager` | class | `memory_core/manager.py` | 会话、提取、检索、导入导出 |
| `MemoryConfig` | dataclass | `config.py` | 全局配置 |
| `ConfigPresets` | class | `config.py` | `minimal/balanced/full_featured` |
| `QwenClient` | class | `memory_core/llm_client.py` | Qwen 聊天、embedding、rerank |
| `MemoryExtractor` | class | `memory_core/extractor.py` | 从对话抽取 `Episode/Fact/UserProfile` |
| `Episode` | dataclass | `memory_core/models.py` | 情景记忆 |
| `Fact` | dataclass | `memory_core/models.py` | 结构化事实 |
| `UserProfile` | dataclass | `memory_core/models.py` | 用户画像 |
| `SQLiteStorage` | class | `storage/sqlite_storage.py` | SQLite 持久化 |

## CONVENTIONS

### 数据模型

```python
@dataclass
class MyModel:
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyModel": ...
```

### 工厂模式

```python
client = create_llm_client(config)  # 返回 QwenClient
extractor = create_extractor(config, llm_client=client)
```

### 异步接口

- `end_session()` 是异步
- `get_memory_context()` 是异步
- 调用检索时应 `await manager.get_memory_context(...)`

## ANTI-PATTERNS

| 禁止 | 原因 |
|------|------|
| 直接访问 `_working_memory_cache` | 必须优先通过公开 API |
| 绕过 `UserProfile.add_tag()` | 会破坏去重和数量约束 |
| 新模型不实现 `to_dict/from_dict` | 导出导入和存储会失效 |
| 假设系统可离线运行 | 当前版本依赖 Qwen |
| 在检索里把 `Fact` 和 `Episode` 混成同一排序逻辑 | 两者语义粒度不同 |

## COMMANDS

```bash
uv pip install --python .\.venv\Scripts\python.exe -e .
uv pip install --python .\.venv\Scripts\python.exe pytest pytest-asyncio httpx pydantic fastapi uvicorn
pytest tests/test_memory.py -q
python -m uvicorn api.server:app --reload --port 8000
```

## TESTING

当前测试方案：

- `QwenClient` 解析测试
- `Episode.embedding` roundtrip
- `end_session()` 抽取 `Episode/Fact/UserProfile`
- `get_memory_context(query=...)` 语义召回
- `Fact` 关键词召回
- 缺少 API Key 报错
- Qwen 异常不降级

## ENVIRONMENT

```bash
set QWEN_API_KEY=your-api-key
```

或：

```bash
set LLM_API_KEY=your-api-key
```
