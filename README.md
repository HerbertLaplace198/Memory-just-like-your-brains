# Neural Memory 1.0：可审计的分层本地记忆系统

系统将记忆正文保存在 Markdown 中，使用 SQLite 神经网络负责激活、联想和分层路由。它独立于 mdkb；默认哈希模式零依赖，也可连接仅限本机的神经语义模型。

## 架构

```text
L6 元记忆域       是否知道、去哪里找
L5 稳定模型       用户画像、目标、稳定偏好
L4 程序记忆       技能、SOP、工作流
L3 语义概念       项目、人物、主题、Wiki关系
L2 情景记忆       一次任务、事件和连续经历
L1 原子记忆       事实、决定、偏好和约束
L0 原始证据       对话、文件、网页和工具输出
```

主要能力：

- 可插拔 `TextEncoder`，默认使用1024维本地稀疏分布式表示；
- 中文二元/三元词片段、BM25、向量相似度融合检索；
- Winner-Take-All 与两轮扩散激活；
- 显式双向突触和可选 Hebbian 强化；
- `proposed / confirmed / rejected / stale / archived` 状态；
- Markdown 权威记录，SQLite 可完整重建；
- Obsidian 连续阅读视图，不作为记忆碎片回流；
- `.nmem` 独立便携包与 SHA-256 完整性校验；
- 固定问题集评测门控准确率和 Top-k 召回。
- 维护 Inbox、有效期、冲突候选和人工确认的替代关系。
- Obsidian 人工批注到候选变更的审核闭环。
- 独立 MCP stdio 服务与受限的 Agent 写入接口。
- 平台无关的任务开始/结束生命周期桥接器。
- WAL、跨进程写锁、原子备份、校验与暂存恢复。
- 仅限本机的 Ollama/OpenAI-compatible 语义编码器与事务迁移。

## 快速演示

```bash
python3 neural_memory.py --root ./demo-memory seed-demo

# 第一层：只判断有没有相关记忆
python3 neural_memory.py --root ./demo-memory probe "怎么减少 Token 消耗"

# 第二层：读取少量记忆卡片
python3 neural_memory.py --root ./demo-memory \
  recall "怎么减少 Token 消耗" --limit 3

# 第三层：沿证据指针读取原始 Markdown
python3 neural_memory.py --root ./demo-memory \
  recall "怎么减少 Token 消耗" --limit 1 --detail
```

判定为 `UNKNOWN` 时，召回门默认关闭，不注入低分候选。调试时可以添加 `--force`。

查看一次召回的完整评分来源：

```bash
python3 neural_memory.py --root ./demo-memory \
  explain "怎么减少 Token 消耗" --limit 10
```

输出会拆分 `vector / bm25 / lexical / direct / spread`。总分由直接检索分、记忆置信度与重要性、以及突触扩散共同组成，便于人工监督误召回。

## 可替换本地编码器

检索器通过 `TextEncoder` 协议接入编码器。默认 `HashEncoder` 不联网、不下载模型，也不需要第三方包。以后可以在 Python 集成层注入本地 ONNX、MLX 或其他 embedding 实现；编码器只需暴露 `name`、`dimensions` 和 `encode(text)`。

同一索引禁止混用不同编码器或维度，避免向量静默损坏。更换编码器时应从 Markdown 权威记录重建到一个新索引，再通过评测集比较后切换。

## 监督与维护

扫描不会自动修改记忆，只会生成可审计的维护候选：

```bash
python3 neural_memory.py --root ./demo-memory maintenance scan
python3 neural_memory.py --root ./demo-memory maintenance inbox
```

写入有期限的事实，或声明新旧关系：

```bash
python3 neural_memory.py --root ./demo-memory remember \
  "用户现在采用新版记忆维护流程" --confirmed \
  --expires "2027-01-01" \
  --supersedes "l1_旧记忆ID"
```

`supersedes` 和 `conflicts` 初始都是 `pending`。只有人工确认替代关系，旧记忆才会变成 `archived`：

```bash
python3 neural_memory.py --root ./demo-memory \
  maintenance confirm-relation "rel_关系ID"

python3 neural_memory.py --root ./demo-memory \
  maintenance resolve "issue_问题ID"
```

疑似冲突检测采用保守的词面重叠与肯定/否定线索，只负责提醒，绝不自动判定哪条为真。到期记忆也不会自动删除；用户可核实后运行 `review stale <ID>`，或写入新版并确认替代关系。

## Obsidian 批注审核闭环

主题页中的 `USER-NOTES` 是人工维护区，但仍然属于输出层。同步操作只创建候选，不会直接摄入：

```bash
python3 neural_memory.py --root ./demo-memory sync-obsidian
python3 neural_memory.py --root ./demo-memory obsidian-review list
python3 neural_memory.py --root ./demo-memory obsidian-review show "note_候选ID"
```

明确接受后，系统才创建一条已确认的 L1 原子记忆，并记录 `obsidian-review:<页面>` 来源：

```bash
python3 neural_memory.py --root ./demo-memory \
  obsidian-review accept "note_候选ID"

python3 neural_memory.py --root ./demo-memory \
  obsidian-review reject "note_候选ID"
```

每次 `compile-obsidian` 还会生成 `99 维护中心.md`，集中显示人工批注候选、系统问题和待审核关系。维护中心和所有主题页都带有 `do_not_ingest: true`。

## 连接 Agent：MCP 服务

`mcp_server.py` 是无第三方依赖的 stdio MCP 适配器。服务启动时固定绑定一个记忆库目录，工具调用不能切换到其他路径。

```bash
python3 mcp_server.py --root /absolute/path/my-neural-memory
```

配置模板见 `mcp.json.example`。将其中两个绝对路径替换为实际位置，再添加到支持 MCP 的 Agent 配置。

对 Agent 暴露五个工具：

- `memory_awareness`：第一阶段，只判断是否有相关记忆；
- `memory_recall`：第二阶段，最多返回五张 L1 卡片，详情默认关闭；
- `memory_explain`：仅在错召回调试时拆解评分；
- `memory_propose`：任务结束时提交候选，状态强制为 `proposed`；
- `memory_inbox`：只读维护收件箱。

推荐 Agent 调用策略：

```text
任务可能依赖历史 → memory_awareness
KNOWN              → memory_recall(detail=false)
证据仍然不足        → memory_recall(detail=true)
产生耐久新事实      → memory_propose
最终确认             → 用户在 CLI / Obsidian 维护层完成
```

服务没有“确认记忆”“删除记忆”或“任意文件读取”工具，因此 Agent 无法绕过人工审核。日常独立问题也不需要调用记忆工具，可以进一步减少 Token 和延迟。

## 生命周期 Hook

`lifecycle_hook.py` 提供平台无关的 JSON stdin/stdout 接口。不同 Agent 只需把自己的生命周期事件映射为 `start` 和 `finish`，示例见 `lifecycle.example.json`。

任务开始：

```bash
printf '%s' '{"task":"怎样减少 Token？","event_id":"task-123"}' | \
  python3 lifecycle_hook.py start --root /absolute/path/my-neural-memory
```

输出只包含最多三张摘要和上层路由，不自动读取完整证据。UNKNOWN 时 `context` 为空。

任务结束：

```bash
printf '%s' '{
  "event_id":"task-123",
  "memory_candidates":[
    {"text":"未来任务仍然有用的事实","topics":["项目主题"]}
  ]
}' | python3 lifecycle_hook.py finish --root /absolute/path/my-neural-memory
```

安全规则：

- 整段 transcript/messages 永远不会自动摄入；
- 只有明确放入 `memory_candidates` 的内容才会写入；
- 每个候选强制为 `proposed`；
- 单次最多接收十条候选；
- `event_id + 正文` 形成幂等键，Hook 重试不会重复写入；
- 开始阶段最多注入三张摘要，详细证据仍由 MCP 按需读取。

查看 Hook 与维护状态：

```bash
python3 lifecycle_hook.py status --root /absolute/path/my-neural-memory
```

## 可靠性、备份与恢复

数据库默认启用 SQLite WAL、5 秒 busy timeout 和 `synchronous=FULL`。所有公开写入路径还会获取 `.write.lock`，协调多个 CLI、MCP 和 Hook 进程；锁支持同一操作中的嵌套写入。

健康检查：

```bash
python3 neural_memory.py --root ./demo-memory doctor
```

输出包括 SQLite `integrity_check`、journal mode、缺失证据和维护统计。

创建带轮换的快照备份：

```bash
python3 neural_memory.py --root ./demo-memory \
  backup ./backups --keep 10
```

备份过程使用 SQLite Backup API 得到一致性快照，检查快照数据库，然后生成带 SHA-256 清单的临时包，最后原子替换为正式 `.nmem`。`--keep 10` 会在新备份成功后删除更旧的轮换备份；设为 `0` 表示不清理。

恢复前可以只校验、不解压：

```bash
python3 neural_memory.py verify-bundle ./backups/neural-memory-时间.nmem
```

恢复仍使用原有命令：

```bash
python3 neural_memory.py --root ./restored-memory \
  import-bundle ./backups/neural-memory-时间.nmem
```

恢复会先在同一磁盘的暂存目录中完成校验与 SQLite 完整性检查，成功后再一次性发布到目标目录。损坏包、路径穿越或中途失败不会形成可见的半恢复记忆库。

## 真正的本地语义编码器

默认仍使用无需模型的 `feature-hash-v1`。如果本机已经运行 Ollama、llama.cpp 或其他 OpenAI-compatible embedding 服务，可以替换为神经网络语义向量。

适配器只允许以下主机：

```text
127.0.0.1
localhost
::1
```

非回环地址会直接拒绝，因此配置不会意外把记忆正文发送到云端。模板：

- `encoder.ollama.example.json`
- `encoder.openai-local.example.json`
- `encoder.hash-256.example.json`（不需要模型，用于测试迁移）

先复制并修改模板中的模型名称、端点和实际向量维度，然后执行：

```bash
python3 neural_memory.py --root ./demo-memory \
  reencode ./encoder.ollama.json
```

迁移过程会：

1. 获取跨进程写锁；
2. 在单个 SQLite 事务内重新编码所有 L1–L6 神经元；
3. 重新构建向量相似关联突触；
4. 成功后保存 `encoder.json`；
5. 任一请求或维度检查失败时整体回滚。

后续 CLI、MCP 和 Hook 会自动读取记忆库中的 `encoder.json`。备份包也会包含该配置。配置不保存 API Key，并且当前 HTTP 适配器仅支持本机无鉴权服务。

建议在正式切换模型前先备份，并使用固定评测集对比旧编码器和新编码器。模型维度必须填写准确，否则迁移会被拒绝。

## 写入 L0–L6

```bash
python3 neural_memory.py --root ./demo-memory remember \
  "重要记忆必须经过人工确认" \
  --source manual \
  --episode "记忆系统设计会话" \
  --topic "记忆治理" \
  --procedure "记忆审核流程" \
  --schema "偏好可监督的系统" \
  --domain "AI记忆与知识管理" \
  --confirmed
```

参数对应：

```text
正文 → L1
--episode → L2
--topic → L3
--procedure → L4
--schema → L5
--domain → L6
```

## Markdown 重建

`vault/memories/*.md` 是权威原子记忆，`vault/evidence/*.md` 是原始证据；SQLite 是可重建运行索引：

```bash
python3 neural_memory.py --root ./demo-memory rebuild
```

审核操作会同时更新 SQLite 和对应的权威 Markdown，避免重建后状态回退。

## 生成 Obsidian 阅读视图

```bash
python3 neural_memory.py --root ./demo-memory compile-obsidian
```

输出位于：

```text
obsidian-view/
├── 00 首页.md
└── 主题/*.md
```

每个页面包含：

- 连续叙述，而不是原子记忆列表；
- 来源记忆 ID 和来源；
- 上层关系；
- `USER-NOTES` 人工批注区；
- `generated: true` 和 `do_not_ingest: true`。

重新编译会更新生成区，但保留人工批注。Obsidian 页面不会自动成为新的记忆碎片。

## 评测

```bash
python3 neural_memory.py --root ./demo-memory \
  evaluate ./evaluation.json --limit 3

python3 -m unittest -v test_neural_memory.py
```

评测报告包含：

- KNOWN/UNKNOWN 门控准确率；
- Top-1 命中率；
- Top-3 召回率；
- 每个问题的峰值激活和实际卡片。

`benchmark` 可以粗略比较全库读取与渐进召回的 Token，但不代表具体模型的正式分词账单：

```bash
python3 neural_memory.py --root ./demo-memory \
  benchmark "怎么减少 Token 消耗" --limit 3
```

## 独立复制与恢复

```bash
python3 neural_memory.py --root ./demo-memory \
  export-bundle ./my-memory.nmem

python3 neural_memory.py --root ./restored-memory \
  import-bundle ./my-memory.nmem
```

`.nmem` 是普通 ZIP，包含数据库、Markdown 权威记录、原始证据、Obsidian 阅读视图、格式清单和每个文件的 SHA-256。导入只允许写入空目录。

## 可选 mdkb 影子适配器

mdkb 不是运行依赖。影子模式仅用于对比：保存标题、标签、派生向量和 `mdkb:<id>` 指针，不复制完整正文。

```bash
python3 neural_memory.py --root ./mdkb-shadow import-mdkb \
  --workspace "/ABSOLUTE/PATH/legacy-mdkb-workspace"
```

## 数据目录

```text
demo-memory/
├── encoder.json            # 可选本地语义编码器配置
├── memory.sqlite3          # 可重建神经索引
├── vault/
│   ├── memories/*.md       # 权威原子记忆
│   └── evidence/*.md       # 原始证据
└── obsidian-view/          # 可重建的人类阅读输出
```

## 已知边界

- 不会让 LLM 自动把整段对话写成长期记忆；这是刻意保留的人工审核边界；
- 冲突检测是候选提醒，不替代用户判断事实真伪；
- 审核界面目前是 CLI 与 Obsidian 维护中心，没有独立 GUI；
- launchd 自动备份提供模板，但需要用户确定最终安装路径后启用；
- Markdown 正文为了 Obsidian 可读而不做应用层静态加密，应使用 FileVault 或加密卷；
- 小型内置评测只用于回归，长期运行应不断加入真实改写和 UNKNOWN 样本重新校准阈值。

完整验收与真实模型 A/B 数据见 `FINAL_REPORT.md`。
