# GrowRAG

GrowRAG 已确认方案 B「可信自适应查询修复」。当前目标是在不修改基础 RAG 的前提下，建立一个可绕过、可追溯、可拒绝历史经验的轻量修复闭环：

> 原查询先运行一次基础 RAG；若证据不足，则显式说明缺口，在可信历史修复与本题现场修复之间选择；没有进展或预算耗尽时停止并承认证据不足。

它可以发展成轻量级单控制器 Agentic RAG，但“Agentic、adaptive、插件式或 query rewriting”都不是项目的既定创新。第一阶段要测量单 Query repair episode 是否真的能跨题迁移，以及能否控制 `BASE 原本正确、REUSE 反而错误` 的非对称伤害。

## 当前入口

- [项目当前记忆](knowledge/CURRENT_PROJECT_MEMORY.md)
- [方案 B 已确认决策](knowledge/decisions/2026-08-22_方案B可信自适应查询修复_已确认.md)
- [单 Query Episode 与长期经验卡 Schema v0](knowledge/method/2026-08-22_单Query_Episode与长期经验卡_Schema_v0.md)
- [Adaptive / Agentic 路由与新颖性边界](knowledge/literature/2026-08-22_方案B自适应路由与新颖性边界.md)
- [方案 B 实施路线](knowledge/experiments/2026-08-22_方案B实施路线.md)
- [双层记忆与结构化修复讨论稿](knowledge/method/2026-08-21_双层记忆与结构化修复_讨论稿.md)
- [历史路线 A 决策](knowledge/decisions/2026-08-19_路线A查询侧可信复用层_已确认.md)
- [查询侧可信复用层 v1](knowledge/method/2026-08-19_查询侧可信复用层_v1.md)
- [查询变体、经验卡、生命周期与双路线](knowledge/method/2026-08-19_查询变体_经验卡_生命周期与双路线.md)
- [v1.1 代码审阅清单](knowledge/method/2026-08-19_代码审阅清单.md)
- [Gate 1 代码实验计划](knowledge/experiments/2026-08-19_Gate1_代码实验计划.md)
- [查询侧可信复用层实施路线](knowledge/experiments/2026-08-19_查询侧可信复用层_实施路线.md)
- [方向候选与待选择事项](knowledge/decisions/2026-08-19_ERM继承路线与待选择.md)
- [ERM 继承与安全复用方案](knowledge/method/2026-08-19_ERM继承与安全复用.md)
- [经验记忆与 EMA 备忘](knowledge/method/2026-08-19_经验记忆与EMA.md)
- [Gold 数据集与阶段规划](knowledge/datasets/2026-08-19_Gold数据集与阶段规划.md)
- [实验阶段与止损门](knowledge/experiments/2026-08-19_实验阶段与止损门.md)
- [当前 Zotero 导入包](zotero/current/README.md)
- [旧工程归档说明](archive/legacy_2026-08-19/MANIFEST.md)

## 代码状态

`src/growrag` 是 2026-08-19 新建的最小研究脚手架。旧版 GrowRAG 已退出活动工程并归档。当前代码保留可靠历史复用子模块，并新增方案 B 的不可变 QueryEpisode、配对验证和长期 ExperienceCard 数据合同；尚未接入真实检索器和 LLM，因此还不是可运行的完整 Agentic 修复闭环。

当前第一批实现包括：

- `DIRECT / REUSE / FRESH` 三种实验动作；
- 来源查询变换、目标题 query plan 与环境版本的数据结构；
- 成对比较 `DIRECT` 与 `REUSE` 的 benefit / harm 标签；
- candidate / active / quarantine / retired 经验账本；
- 严格、可重启的经验快照和运行配置；
- 历史候选召回、当前适用性、可信 gate 和完整 facade；
- S2G 启发的来源缺口经验卡、正向/禁用适用条件；
- 非破坏性的衰减热记忆 Top-K 与 strict/tiered 环境兼容；
- 可扫描阈值的 risk–coverage 离线分析；
- 文件化批量决策 CLI 与离线 oracle；
- 可复现研究所需的防泄漏与风险测试。
- 强制 BASE-first 的单 Query episode 状态机；
- 每个 turn 单查询、结构化 gap、证据指针、成本与停止原因；
- 同题、同环境、同预算的 BASE/REUSE/FRESH 配对验证；
- typed episode/turn provenance、版本父链与不可覆盖卡片；
- 可重算 benefit/neutral/correct-to-wrong harm 的经验卡注册边界；
- 可配置但显式记录的 ACTIVE 门槛，默认拒绝 proxy-only 和单次成功。

旧 v1 facade 的输出仍是“下游应执行的原查询或复用查询”；新的 episode/card 合同目前尚未接到旧 facade，避免两套生命周期被误当成已经整合。下一阶段是接入一个冻结的 BM25 BASE、结构化 state judge 与 FRESH fallback，再让旧 reliability/applicability gate 通过新版 Registry 消费卡片。EMA、RL/bandit 仍不属于首版。

## 本地启动

推荐 Python 3.11 或 3.12。通用安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux/macOS 将第二行替换为：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

先运行无需模型的内置示例：

```powershell
python -m growrag demo
```

再运行完整的文件化 v1：

```powershell
python -m growrag config-validate --config examples/trusted_reuse/config.toml
python -m growrag memory-validate --memory examples/trusted_reuse/memory.json
python -m growrag decide `
  --config examples/trusted_reuse/config.toml `
  --memory examples/trusted_reuse/memory.json `
  --requests examples/trusted_reuse/requests.jsonl `
  --plans examples/trusted_reuse/plans.json `
  --output runs/trusted-reuse-demo.jsonl
```

四个输入分别是：

- `config.toml`：冻结的环境、风险门和预算；
- `memory.json`：带 schema 和快照 ID 的可信经验库；
- `requests.jsonl`：当前问题及检索前结构标签；
- `plans.json`：每个“当前问题 × 历史经验”的预生成查询方案。

输出逐题记录 `DIRECT/REUSE`、最终查询、候选召回、可靠度与适用性检查，但不包含目标题 gold、答案或检索结果。

离线 Oracle 示例：

```powershell
python -m growrag oracle --input examples/oracle_pilot.csv --output runs/oracle_pilot.json
```

验证仓库：

```powershell
python -m pytest
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

## 仓库边界

- `data/`、`runs/` 和 `external/` 的大文件不提交 Git。
- `archive/` 只跟踪说明文件，不提交旧代码、PDF 或历史运行产物。
- GitHub 远端仓库已创建且本地 `origin` 已配置；当前命令行网络无法连接 GitHub，连接器仍待授权，因此尚未完成首次推送。见 [GITHUB_SETUP.md](GITHUB_SETUP.md)。
