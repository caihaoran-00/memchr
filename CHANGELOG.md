# Changelog

## [Unreleased] - 2026-03-12

### Added
- 新增 `QwenClient`，统一支持聊天抽取、embedding 和 rerank。
- 新增项目级 `.env` 读取和 `.env.example`。
- 新增统一单文件配置入口 [memory_settings.json](/E:/chr_git/memchr/memory_settings.json) 和模板 [memory_settings.example.json](/E:/chr_git/memchr/memory_settings.example.json)。
- 新增 [tests/self_check.py](/E:/chr_git/memchr/tests/self_check.py) 配置自检脚本。
- 新增更适合直接体验的 [tests/demo.py](/E:/chr_git/memchr/tests/demo.py)。
- 新增项目根目录一键入口 [run_demo.py](/E:/chr_git/memchr/run_demo.py)。
- 新增可配置 provider 组合：`llm_provider`、`embedding_provider`、`rerank_provider`。
- 新增 `Episode` 语义检索链路：`embedding -> candidate recall -> rerank -> final ranking`。
- 新增 `episodes.embedding` 的写入、读取和测试覆盖。
- 新增基于 Qwen stub 的完整集成测试。

### Changed
- 记忆提取默认使用 Qwen，但可通过配置切换为 `deepseek`。
- 项目内不再保留 `MockLLMClient` / `mock provider`。
- 绝大多数项目配置现在都可以在 `memory_settings.json` 中集中修改，不必再改代码。
- 结构整理：`config.py` 的项目配置回填改为 helper 统一处理，`llm_client.py` 的 Qwen 客户端构建收敛为单入口，`manager.py` 的 Fact 查询扩展拆成更清晰的辅助函数。
- `Fact` 检索增加自然语言查询扩展，真实问句召回更稳。
- `MemoryManager.get_memory_context()` 改为异步，并切换为 `Episode` 语义召回、`Fact` 关键词召回。
- `README.md`、`AGENTS.md`、`memory_core/AGENTS.md` 更新为 Qwen-only 方案说明。
- 数据模型和 `system_prompt` 组装逻辑清理为可读实现。

### Fixed
- 修复 `import_user_memory()` 跨用户导入时 `episodes/facts` 归属错误的问题。
- 修复导入到新用户时 `Episode` / `Fact` 主键冲突的问题。
- 修复 `PUT /profile` 更新 `tags` 时绕过 `add_tag()` 约束的问题。
- 修复包导出层仍引用 `RuleBasedExtractor` 的残留问题。

### Removed
- 删除规则提取兜底路径。
- 删除项目内 mock provider 兜底路径。
