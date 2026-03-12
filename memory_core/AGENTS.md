# memory_core AGENTS

## OVERVIEW

`memory_core` 是项目的核心逻辑层，负责：

- 数据模型定义
- Qwen 记忆提取
- `Episode` 语义检索
- `Fact` 关键词检索
- `MemoryContext` 组装

当前版本不再保留规则提取和项目内 mock provider。

## WHERE TO LOOK

| 任务 | 文件 | 说明 |
|------|------|------|
| 数据模型 | `models.py` | `Message / WorkingMemory / Episode / Fact / UserProfile / MemoryContext` |
| 提取逻辑 | `extractor.py` | Qwen JSON 抽取 |
| 模型客户端 | `llm_client.py` | `QwenClient` |
| 检索与会话 | `manager.py` | 召回、排序、导入导出 |

## KEY FLOW

```text
start_session()
  -> add_message()
  -> end_session()
     -> Qwen extract_json()
     -> create Episode / Fact / UserProfile
     -> save to SQLite

get_memory_context(query)
  -> load profile
  -> query embedding
  -> episode semantic recall
  -> Qwen rerank
  -> fact keyword search
  -> build MemoryContext
```

## NOTES

- `Episode` 是摘要型情景记忆，适合语义检索
- `Fact` 是结构化知识点，当前保持关键词检索
- `UserProfile` 是长期聚合画像，不参与相似度检索
- `get_memory_context()` 和 `end_session()` 都是异步
