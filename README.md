# GrowRAG

GrowRAG 已确认第一版研究方向，并完成可安装、可配置、可重启的 v1 MVP；下一步进入真实数据可行性验证：

> 在第一次正式检索前，只有当一条历史查询变换本身可靠、且对当前问题的适用性风险足够低时才复用；否则保持原查询不动。

这不是“新 RAG 范式”的既定结论。第一阶段先测量历史查询变换是否真的存在跨题迁移空间，以及能否控制 `DIRECT 原本正确、REUSE 反而错误` 的非对称伤害。

## 当前入口

- [项目当前记忆](knowledge/CURRENT_PROJECT_MEMORY.md)
- [路线 A 已确认决策](knowledge/decisions/2026-08-19_路线A查询侧可信复用层_已确认.md)
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

`src/growrag` 是 2026-08-19 新建的最小研究脚手架。旧版“账本—验证门—经验注入”代码已经退出活动工程，保存在 `archive/legacy_2026-08-19`，不得再作为新方法的默认依据。

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

v1 的输出是“下游应执行的原查询或复用查询”，尚未接入真实检索器、LLM 和公开数据。ERM 索引更新、EMA、RL/bandit 也不属于当前 v1。

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
- GitHub 远端尚未创建；原因和下一步见 [GITHUB_SETUP.md](GITHUB_SETUP.md)。
