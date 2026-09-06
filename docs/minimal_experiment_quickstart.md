# 最小配对骨架：现在能做什么

状态：2026-09-06。已有离线示例与真实 API 单步配对执行器。真实结果以具体 run 的审计与汇总为准，不能用单元测试或 mock 代替。

## 当前真实实验入口

先读 [架构 v1](architecture_v1.md) 与 [数据协议](../knowledge/datasets/2026-09-06_正式数据协议与阶段划分.md)。普通百炼配置只保存在已忽略的本地 `qwenAPI.md`，不能推送。

首次下载公开 train 的前 200 条完整记录，冻结划分和 8＋8 题号（已存在时不要重复覆盖）：

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments.data_protocol --output data/hotpotqa/train_preview_200_v1 --allow-download
```

只有获准付费试跑时才执行下一命令；输出目录必须新建，命令会真实调用 API：

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments.run_pilot --manifest data/hotpotqa/train_preview_200_v1/manifest.json --api-config qwenAPI.md --output runs/2026-09-06_hotpot_pilot_v1 --allow-network
```

固定 `qwen3.7-flash-2026-07-15`、关闭思考；最多 96 次请求、每次最多 768 输出 tokens、应用估算预算 1 元、时间窗口 15 分钟。不自动重试、不自动换模型；用量未知或预算异常后停止联网。金额限制基于北京定价与保守预留，不是厂商账单硬封顶承诺，最终以供应商账单为准。

输出：`launch_plan.json` 保存版本和输入；`episodes/` 保存每题；`memory_library.json` 是来源完成后冻结的候选库；`pilot_report.json` 含效果与失败；`budget_report.json` 含全部调用成本；`api_audit/` 含真实请求响应。全部保留在本地、不提交 Git。

## 看一次不收费的流程演示

在项目根目录运行（输出目录必须是新的）：

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments --output runs/my-first-mock
```

该命令不会联网调用模型。例子的资料、缺口和查询都是手写的测试输入；它只测试线路，不能说明经验有效。终端和结果均标明 MOCK ONLY。

本轮已经生成的文件：`runs/2026-09-06_single_repair_mock_v1/run.json`。

回看现有结果，不重复执行任何组件：

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments --replay runs/2026-09-06_single_repair_mock_v1/run.json
```

## 你审阅时只看六件事

1. `state`：三条路线是否从同一个问题和同一份已有证据出发？
2. `memory`：FRESH 为 null，REUSE 才有来源经验；源问题不能是当前问题。
3. `query`：各自下一步查什么，最终 Reader 仍回答原问题。
4. `new_evidence`：新增哪些原文证据；三条路线不共享后续证据。
5. `calls`：成功、失败、实际已知用量与审计路径；未知 token 是 null，不是 0。
6. `execution_kind`：mock 不算模型结果；real 还要检查 API 请求/响应审计，不能单凭一个字段证明真实性。

## 新增模块

- `src/growrag/experiments/protocol.py`：输入与记录结构。
- `src/growrag/experiments/paired_runner.py`：同状态、单次修复的分支执行。
- `src/growrag/experiments/hotpot.py`：本地 Hotpot 格式读取和独立证据/答案反馈；不会下载数据。
- `src/growrag/experiments/api_client.py`：显式启用的 HTTPS 调用与脱敏审计；默认禁止发送。
- `src/growrag/experiments/llm_adapters.py`：有版本的查询改写和回答提示词，真实 API 输出须通过格式/引用检查。

基础 `paired_runner` 接收外部 gap 和 MemoryView；上层 `pilot_engine` 负责真实首检、固定 prompt 的 gap、来源笔记生成和冻结。底层分支费用只覆盖分支，上层报告与 budget 报告覆盖 gap、来源建库及全部调用，不能只挑目标分支成本展示。当前 gap 是代理判断，并未训练或复现 S2G Judge。

## API 安全与可复现边界

真实 API 必须显式指定 endpoint、模型、密钥环境变量名、最大调用数和输出上限，并启用网络。没有后台默认服务、没有借用聊天登录令牌、没有自动换模型/重试/补写答案。

请求和响应审计文件含模型、prompt 版本、时间、usage、响应 ID、哈希与失败状态。Authorization 不入日志，已知密钥回显会脱敏并拒绝作为正常结果。文件可能包含公开题目与证据，仍不要把私人数据随意提交 Git；运行目录已忽略。

测试替身必须声明 mock；缺失来源或与返回来源不符会被拒绝。测试里对 HTTP 的替换只是单元测试，不是 live 记录。缺失计量不自行估算成实测值。次数/输出上限不是完整的金额限额，真正运行前还要配置供应商预算。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src/growrag/experiments tests/test_experiment*.py
```

此前骨架：256 项测试通过。后续 API/BM25 更新后全套 304 项通过，静态检查通过。它们证明已覆盖的代码行为，不证明论文效果。2026-09-06 已真实运行一次固定 Flash 连通探针（23 输入/5 输出 tokens），没有运行 HotpotQA 配对；详见[API 核验记录](../knowledge/decisions/2026-09-06_TokenPlan与按量API_首次真实连通检查.md)。
