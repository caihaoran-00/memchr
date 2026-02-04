# 智能玩具记忆系统

为儿童智能对话玩具设计的轻量级记忆系统，支持短期和长期记忆管理。

---

## 整体设计思路

### 设计目标

为 **ASR + LLM + TTS** 方案的智能对话玩具增加记忆能力，让玩具能够：
- 记住孩子的名字、年龄、喜好等基本信息
- 回忆之前聊过的话题和重要事件
- 随着时间推移自然地"遗忘"不重要的内容
- 在成本可控的前提下提供良好的记忆效果

### 核心设计理念

```
┌─────────────────────────────────────────────────────────────────────┐
│                     人类记忆模型 → 系统设计                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   人脑记忆                        系统实现                          │
│   ────────                        ────────                          │
│   感觉记忆 (毫秒级)     →         (不需要，ASR已处理)               │
│   短期记忆 (秒~分钟)    →         工作记忆 (当前对话上下文)          │
│   长期记忆 - 情景记忆   →         情景记忆 (对话摘要/事件)           │
│   长期记忆 - 语义记忆   →         语义记忆 (用户画像/知识事实)       │
│   遗忘曲线             →         遗忘机制 (时间衰减+重要性)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三层记忆架构详解

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Memory System                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│  │   工作记忆     │  │   情景记忆     │  │      语义记忆           │  │
│  │  (Working)    │  │  (Episodic)   │  │     (Semantic)          │  │
│  ├───────────────┤  ├───────────────┤  ├─────────────────────────┤  │
│  │ • 当前对话    │  │ • 对话摘要    │  │ • 用户画像              │  │
│  │ • 最近N轮     │  │ • 重要事件    │  │   - 名字/年龄/性别      │  │
│  │ • 滑动窗口    │  │ • 情感记录    │  │   - 兴趣标签            │  │
│  │ • 内存存储    │  │ • 关键词索引  │  │ • 知识事实              │  │
│  │              │  │              │  │   - 三元组(主谓宾)       │  │
│  └───────────────┘  └───────────────┘  └─────────────────────────┘  │
│         │                  │                      │                 │
│         ▼                  ▼                      ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Memory Manager (记忆管理器)               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │  提取    │ │  压缩    │ │  遗忘    │ │  检索    │       │   │
│  │  │ Extract  │ │ Compress │ │  Forget  │ │ Retrieve │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Storage Layer (存储层)                       │   │
│  │          SQLite (轻量级，适合嵌入式设备)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 第一层：工作记忆 (Working Memory)

**类比人脑**：类似于人类的短期记忆，保持当前正在进行的对话上下文。

**设计特点**：
- 使用**滑动窗口**机制，只保留最近N轮对话
- 存储在**内存**中，快速读写
- 会话结束时压缩为情景记忆

**数据流**：
```
用户说话 → ASR转文字 → 加入工作记忆 → LLM生成回复 → 回复加入工作记忆
                              ↓
                    (会话结束时压缩)
                              ↓
                         情景记忆
```

### 第二层：情景记忆 (Episodic Memory)

**类比人脑**：类似于人类记住的具体事件和经历，如"上次聊过恐龙的话题"。

**设计特点**：
- 存储**对话摘要**而非原始对话（节省空间）
- 附带**关键词**索引，便于检索
- 记录**情感标签**和**重要性评分**
- 支持**时间衰减**，模拟自然遗忘

**数据结构**：
```python
Episode:
  - summary: "聊了关于恐龙的话题，小明说他最喜欢霸王龙"
  - keywords: ["恐龙", "霸王龙", "喜欢"]
  - emotion: "开心"
  - importance: 0.8  # 0-1，越高越不容易被遗忘
  - access_count: 3  # 被检索次数，越多越不容易被遗忘
  - created_at: "2024-01-15"
  - last_accessed: "2024-01-20"
```

### 第三层：语义记忆 (Semantic Memory)

**类比人脑**：类似于人类的知识和常识，如"小明喜欢恐龙"这种抽象事实。

**包含两部分**：

#### 3.1 用户画像 (User Profile)
```python
UserProfile:
  - name: "小明"
  - age: 5
  - gender: "男"
  - tags: ["喜欢恐龙", "害怕打雷", "有个朋友叫小红"]
```

#### 3.2 知识事实 (Facts)
使用**三元组**存储结构化知识：
```python
Fact:
  - subject: "小明"      # 主语
  - predicate: "喜欢"    # 谓语
  - object: "恐龙"       # 宾语
  - confidence: 0.9      # 置信度
```

---

## 数据流程

### 完整对话流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         对话开始                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. 开始会话 (start_session)                                        │
│     - 创建工作记忆                                                   │
│     - 加载用户画像（如果存在）                                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. 用户说话                                                         │
│     ASR → "我叫小明，我今年5岁了"                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. 记录消息 (add_message)                                          │
│     - 消息加入工作记忆                                               │
│     - 滑动窗口保持最近N轮                                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. 获取记忆上下文 (get_memory_context)                              │
│     - 检索相关情景记忆                                               │
│     - 检索相关知识事实                                               │
│     - 获取用户画像                                                   │
│     - 组装成系统提示词                                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. 调用LLM生成回复                                                  │
│     System Prompt:                                                   │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │ 【用户信息】                                             │     │
│     │ 用户名字：小明                                           │     │
│     │ 年龄：5岁                                                │     │
│     │ 兴趣特征：喜欢恐龙, 害怕打雷                              │     │
│     │                                                         │     │
│     │ 【已知信息】                                             │     │
│     │ - 小明喜欢霸王龙                                         │     │
│     │ - 小明的好朋友是小红                                     │     │
│     │                                                         │     │
│     │ 【相关记忆】                                             │     │
│     │ - 上次聊过恐龙的话题                                     │     │
│     └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. 记录助手回复                                                     │
│     - 回复加入工作记忆                                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. 结束会话 (end_session)                                          │
│     - 使用LLM/规则提取关键信息                                       │
│     - 生成情景记忆（对话摘要）                                        │
│     - 提取知识事实（三元组）                                          │
│     - 更新用户画像（标签/属性）                                       │
│     - 清理工作记忆                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 配置项详解

### 存储配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `data_dir` | `"./data"` | 数据存储目录路径 |
| `db_name` | `"memory.db"` | SQLite数据库文件名 |

### 工作记忆配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `working_memory_size` | `10` | 工作记忆保留的最大对话轮数（滑动窗口大小）。设置为10表示保留最近10轮对话（20条消息）。增大可提供更多上下文，但会增加LLM token消耗 |

### 情景记忆配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_episodes_per_user` | `100` | 每个用户最多保存的情景记忆数量。超出时自动删除最旧且最不重要的记忆。控制存储空间和检索效率 |
| `episode_summary_max_length` | `200` | 情景摘要的最大字符长度。限制单条记忆的大小，避免摘要过长 |
| `episode_compress_threshold` | `5` | 触发记忆提取的最小对话轮数。少于5轮的对话被认为太短，不值得保存为情景记忆 |

### 语义记忆配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_profile_tags` | `20` | 用户画像最多保留的兴趣/特征标签数。新标签会挤掉最旧的标签 |
| `max_facts_per_user` | `50` | 每个用户最多保存的知识事实数量。超出时删除置信度最低的事实 |

### 遗忘机制配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `memory_decay_days` | `30` | 记忆完全衰减的天数。30天没有被访问的记忆，时间衰减因子降为0 |
| `min_importance_threshold` | `0.2` | 记忆强度阈值。低于此值的记忆将被自动清理（遗忘） |
| `access_count_weight` | `0.3` | 访问次数在记忆强度计算中的权重。经常被检索到的记忆更不容易被遗忘 |
| `time_decay_weight` | `0.7` | 时间衰减在记忆强度计算中的权重。时间越久远记忆越容易被遗忘 |

**记忆强度计算公式**：
```
strength = importance × (time_decay_weight × 时间因子 + access_count_weight × 访问因子)

时间因子 = max(0, 1 - 距离上次访问天数 / memory_decay_days)
访问因子 = min(1, access_count / 10)
```

### LLM配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm_provider` | `"openai"` | LLM服务提供商。可选：`"openai"`, `"zhipu"`(智谱), `"mock"`(测试用) |
| `llm_api_key` | 环境变量 | API密钥，从环境变量 `LLM_API_KEY` 读取 |
| `llm_base_url` | 环境变量 | API基础URL，可用于配置代理或兼容接口 |
| `llm_model` | `"gpt-4o-mini"` | 对话使用的模型。推荐使用低成本模型 |
| `extraction_model` | `"gpt-4o-mini"` | 记忆提取使用的模型。可以用更便宜的模型 |
| `llm_max_retries` | `3` | API调用失败时的最大重试次数 |
| `llm_timeout` | `30` | API请求超时时间（秒） |

### 向量检索配置（可选功能）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_vector_search` | `False` | 是否启用向量语义检索。关闭可降低成本，使用关键词检索替代 |
| `vector_dim` | `384` | 向量维度。使用小模型降低计算成本 |
| `embedding_model` | `"sentence-transformers/..."` | 本地embedding模型。使用多语言小模型 |
| `similarity_threshold` | `0.7` | 向量相似度阈值。低于此值的结果不返回 |
| `max_retrieval_results` | `5` | 检索返回的最大结果数 |

### 成本控制配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `batch_size` | `5` | 批量处理大小。累积多个请求后一起处理，减少API调用次数 |
| `cache_ttl` | `3600` | 缓存过期时间（秒）。1小时内的重复检索使用缓存 |
| `enable_cache` | `True` | 是否启用本地缓存 |

---

## 预设配置模板

系统提供三种预设配置，适用于不同场景：

### minimal() - 最小配置

```python
ConfigPresets.minimal()
```

| 特点 | 说明 |
|------|------|
| LLM提供商 | Mock（不调用真实API） |
| 工作记忆 | 5轮 |
| 情景记忆 | 最多20条 |
| 向量检索 | 关闭 |
| 适用场景 | **开发测试、功能验证** |
| API成本 | **零** |

### balanced() - 平衡配置

```python
ConfigPresets.balanced()
```

| 特点 | 说明 |
|------|------|
| LLM提供商 | OpenAI |
| 模型 | gpt-4o-mini（低成本） |
| 工作记忆 | 10轮 |
| 情景记忆 | 最多50条 |
| 向量检索 | 关闭（使用关键词检索） |
| 适用场景 | **生产环境、日常使用** |
| API成本 | **低** |

### full_featured() - 完整配置

```python
ConfigPresets.full_featured()
```

| 特点 | 说明 |
|------|------|
| LLM提供商 | OpenAI |
| 模型 | gpt-4o（高质量） |
| 工作记忆 | 15轮 |
| 情景记忆 | 最多100条 |
| 向量检索 | 开启（语义检索） |
| 适用场景 | **高端产品、追求最佳效果** |
| API成本 | **较高** |

---

## 成本控制策略

### 1. 选择合适的配置预设

```python
# 开发测试时
config = ConfigPresets.minimal()  # 零成本

# 生产环境
config = ConfigPresets.balanced()  # 低成本高效果
```

### 2. 使用规则提取替代LLM提取

```python
from memory_core.extractor import RuleBasedExtractor

# 规则提取器不调用LLM，零成本
extractor = RuleBasedExtractor(config)
result = extractor.extract_from_conversation(messages, user_id)
```

### 3. 合理设置记忆阈值

```python
config = MemoryConfig()
config.episode_compress_threshold = 8  # 只有8轮以上对话才提取记忆
config.min_importance_threshold = 0.3  # 提高遗忘阈值，保留更少记忆
```

### 4. 使用便宜的模型

```python
config = MemoryConfig()
config.llm_model = "gpt-4o-mini"        # 对话用便宜模型
config.extraction_model = "gpt-4o-mini"  # 提取也用便宜模型
```

### 5. 关闭向量检索

```python
config = MemoryConfig()
config.enable_vector_search = False  # 使用关键词检索，无需embedding计算
```

---

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# 完整安装（包含API服务和LLM支持）
pip install -e ".[all]"

# 或手动安装依赖
pip install pydantic httpx jieba fastapi uvicorn
```

### 基础使用

```python
import asyncio
from config import ConfigPresets
from memory_core.manager import MemoryManager

async def main():
    # 使用预设配置
    config = ConfigPresets.balanced()
    manager = MemoryManager(config)
    
    # 开始会话
    session = manager.start_session("child_001")
    
    # 添加对话
    manager.add_message(session.session_id, "user", "我叫小明，我5岁了")
    manager.add_message(session.session_id, "assistant", "你好小明！")
    manager.add_message(session.session_id, "user", "我喜欢恐龙")
    manager.add_message(session.session_id, "assistant", "恐龙很酷！")
    
    # 获取记忆上下文（用于增强LLM对话）
    context = manager.get_memory_context(session.session_id)
    print(context.to_system_prompt())
    
    # 结束会话，提取记忆
    episode = await manager.end_session(session.session_id)
    if episode:
        print(f"记忆摘要: {episode.summary}")

asyncio.run(main())
```

### 运行演示

```bash
cd toy_memory_system
python tests/demo.py
```

### 启动API服务

```bash
cd toy_memory_system
python -m uvicorn api.server:app --reload --port 8000
```

API文档：http://localhost:8000/docs

---

## 项目结构

```
toy_memory_system/
├── config.py              # 配置管理（含预设模板）
├── pyproject.toml         # 项目依赖配置
├── README.md              # 本文档
│
├── memory_core/           # 核心记忆模块
│   ├── __init__.py
│   ├── models.py          # 数据模型（Message/Episode/Fact/Profile）
│   ├── manager.py         # 记忆管理器（核心控制层）
│   ├── extractor.py       # 记忆提取器（LLM+规则两种模式）
│   └── llm_client.py      # LLM客户端（支持OpenAI/智谱/Mock）
│
├── storage/               # 存储层
│   ├── __init__.py
│   └── sqlite_storage.py  # SQLite持久化存储
│
├── api/                   # HTTP API服务
│   ├── __init__.py
│   └── server.py          # FastAPI接口
│
├── examples/              # 集成示例
│   ├── __init__.py
│   └── integration_example.py  # 完整集成示例
│
├── tests/                 # 测试
│   ├── __init__.py
│   ├── demo.py            # 演示脚本
│   └── test_memory.py     # 单元测试
│
└── data/                  # 数据目录（运行时自动创建）
```

---

## API 接口

### 会话管理

- `POST /session/start` - 开始会话
- `POST /session/message` - 添加消息
- `POST /session/end` - 结束会话

### 记忆检索

- `POST /context` - 获取记忆上下文

### 用户管理

- `GET /profile/{user_id}` - 获取用户画像
- `PUT /profile` - 更新用户画像
- `GET /stats/{user_id}` - 获取统计信息

### 数据导入导出

- `GET /export/{user_id}` - 导出记忆
- `POST /import` - 导入记忆

### 维护操作

- `POST /maintenance/forget/{user_id}` - 运行遗忘机制
- `POST /maintenance/cleanup` - 清理过期数据

---

## 与玩具主程序集成

```python
from memory_core.manager import MemoryManager
from config import MemoryConfig

class ToyAssistant:
    def __init__(self):
        config = MemoryConfig()
        config.llm_provider = "openai"  # 或 "zhipu" 使用国内API
        self.memory = MemoryManager(config)
    
    async def chat(self, user_id: str, user_input: str) -> str:
        # 获取或创建会话
        session = self.memory.start_session(user_id)
        
        # 添加用户消息
        self.memory.add_message(session.session_id, "user", user_input)
        
        # 获取记忆上下文
        context = self.memory.get_memory_context(session.session_id, user_input)
        
        # 构建增强的prompt
        system_prompt = f"""你是一个可爱的智能玩具助手。
{context.to_system_prompt()}
"""
        
        # 调用LLM获取回复
        response = await self.call_llm(system_prompt, user_input)
        
        # 记录助手回复
        self.memory.add_message(session.session_id, "assistant", response)
        
        return response
```

---

## 环境变量

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"  # 可选，用于代理或兼容API
```

---

## 许可证

MIT License
