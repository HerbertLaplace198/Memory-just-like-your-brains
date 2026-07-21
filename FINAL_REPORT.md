# Neural Memory 1.0 验收报告

## 结论

系统已经从概念 Demo 收口为独立、可复制、可审计的本地记忆系统：L0–L6 分层记忆、三阶段渐进召回、Obsidian 连续阅读输出、人工审核、冲突与时间治理、MCP、生命周期 Hook、并发保护、原子备份恢复，以及真实本地神经语义向量均已实现。

最终默认语义模型为 `qwen3-embedding:0.6b`，通过 Ollama 在 `127.0.0.1` 运行。官方 Ollama 包约 639 MB；模型为 0.6B 参数、1024 维、32K 上下文并支持 100+ 语言。

## A/B 结果

测试库包含 5 条原子记忆、13 个 L1–L6 神经元。语义改写集包含 6 个应召回问题和 6 个无关问题。

| 编码器 | 门控准确率 | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1024维特征哈希 | 83.33% | 66.67% | 66.67% |
| Qwen3-Embedding-0.6B | 100% | 100% | 100% |

原有 8 问词面评测中，Qwen 在阈值校准前 Top-1/Top-3 已为 100%，但 UNKNOWN 门控只有 62.5%。将本地神经编码器门控阈值从哈希的 0.06 单独校准为 0.30 后，门控恢复为 100%。这说明“语义排名”和“我是否真的知道”必须使用分离校准，不能直接套用同一阈值。

这些结果来自小型合成测试，只证明当前回归集通过，不代表开放域质量上限。正式积累记忆后应持续扩展 UNKNOWN 与改写问题集。

## 安全边界

- Embedding HTTP 只允许 `localhost`、`127.0.0.1`、`::1`；环境代理被强制绕过。
- MCP 不能确认、删除记忆或切换任意目录。
- Agent 写入始终是 `proposed`。
- transcript/messages 不会被生命周期 Hook 自动摄入。
- Obsidian 输出带 `do_not_ingest: true`，人工批注必须二次确认。
- 数据正文保持 Markdown 可读，因此应用层未做静态加密；应使用 macOS FileVault 或把记忆库放在加密卷中。

## 运行与恢复

- SQLite WAL、`synchronous=FULL`、跨进程写锁。
- SQLite Backup API 一致性快照、SHA-256 清单、原子发布。
- 暂存恢复完成校验后才切换目标目录。
- 提供每日 03:00、保留 10 份的 launchd 模板，但由于最终记忆库安装路径属于用户选择，发行包不会擅自安装该计划任务。

## 验收记录

- 24 项自动测试通过。
- 8 个独立连接并发写入通过。
- 篡改包检测和拒绝恢复通过。
- 导出、校验、暂存恢复、恢复后健康检查通过。
- Ollama 0.32.1 已通过 Homebrew 安装并作为登录后台服务运行。
- `qwen3-embedding:0.6b` 已下载并完成真实批量 embedding。

## 官方模型资料

- [Ollama：qwen3-embedding](https://ollama.com/library/qwen3-embedding)
- [Qwen3-Embedding-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Ollama Embedding API 示例](https://ollama.com/blog/embedding-models)
