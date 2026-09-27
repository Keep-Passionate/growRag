# Shared500 离线审计工具

这些脚本只读已有实验记录；不调用模型 API、不读取 API 密钥、不训练或更新记忆、不修改原始 run。它们针对固定的 `500_v1` train-development 实验及其 manifest SHA，不是通用 benchmark 工具，也不把开发集结果称为官方测试复现。

在项目根目录运行，先安装本项目及测试依赖。命令中的输出目录必须是新目录，不能覆盖旧结果。

| 工具 | 输入与用途 | 明确边界 |
| --- | --- | --- |
| `audit_shared500.py` | 已封口批次的账本、journal、API 原始审计及报告 | 未知费用不是零；已知费用是 token 估价，不是供应商账单；不重复累计历史预算 |
| `request_profile_audit.py` | 原始API审计中的模型、角色、容量、JSON schema及请求标识 | 不输出消息/响应正文/凭据；本地一致性不是供应商公证；T0不保证确定输出 |
| `trajectory_stats.py` | 已封口批次的执行轨迹、来源快照和评分报告 | 重复 query 不等于无效；分类不是因果分析；失败与完整配对分开 |
| `missing_bounds.py` | 聚合 `analysis.json` 与固定 manifest | 缺失结果的确定性最坏/最好边界，不是补分、置信区间或显著性检验 |
| `query_replay_probe.py` | 无 gold 执行文件、SQLite 索引、固定作者快照；重放冻结后才读 gold | 相同真实状态下比较支持原文到达；无 Reader 调用，不能推断答案收益；需要额外离线 CPU |

```powershell
.venv/Scripts/python.exe scripts/audits/audit_shared500.py --output runs/shared500_api_audit_final
.venv/Scripts/python.exe scripts/audits/request_profile_audit.py --output runs/shared500_request_profile_final.json
.venv/Scripts/python.exe scripts/audits/trajectory_stats.py --output-dir runs/shared500_trajectory_final
.venv/Scripts/python.exe scripts/audits/missing_bounds.py --analysis runs/CHOOSE_AGGREGATE/analysis.json --manifest data/hotpotqa/shared500_sep27_v1/manifest.json
.venv/Scripts/python.exe scripts/audits/query_replay_probe.py --all-closed --workers 2 --output runs/shared500_query_replay_final
```

`audit_shared500.py` 和 `trajectory_stats.py` 可以只分析当前已封口部分，必须查看实际分母。`missing_bounds.py` 也可在未完成阶段运行，此时未启动题同样使界限变宽。

`query_replay_probe.py --all-closed` 必须等固定 500 题全部有工程封口记录；它再核对精确题号与 offset 0–499。失败题仍在工程完成记录中，但不进入完整配对比较。小规模验证使用 `--limit 2`（最多 6），默认 1 个进程；`--workers 2` 是允许的最大值。每个进程单独打开只读 SQLite，输出按 offset/round 排序。逻辑检索次数、索引调用、每题缓存命中分别报告，缓存不是在线方法优势。

## 测试与可选本地资源

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_shared500_api_audit.py tests/test_shared500_query_replay.py tests/test_shared500_trajectory_stats.py tests/test_shared500_missing_bounds.py tests/test_shared500_request_profile.py
```

绝大多数测试只用合成数据，不要求题库、API、密钥或索引。query replay 的两项可选集成测试会使用 `external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6`：若本地快照缺失，明确跳过；已有文件的 SHA 不匹配则失败。测试不会自行下载第三方代码。

仓库只包含我们编写的审计逻辑与合成测试；不提交作者源码快照、数据集、API 响应、原始运行记录或凭据。人工整理的结果报告会入库，原始审计留本地。全量 query replay 需要自行准备并核验上述快照和现有只读索引；其余四个工具不执行作者代码。
