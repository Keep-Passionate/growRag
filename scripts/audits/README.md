# Shared500 离线审计工具

## ReFormeR续轮（2026-09-27）

`analyze_reformer_pilot.py`读取已封口的ReFormeR固定100题迁移记录和旧500题BASE/S2G缓存，核对每次调用归属，并用相同完整题交集比较质量、费用和oracle额外机会。不是原ReFormeR论文TREC复现；未执行/失败不补0，旧缓存不重复计费。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/audits/analyze_reformer_pilot.py --output runs/reformer_new_offline_review
```

已生成当前阅读入口`runs/2026-09-27_reformer_analysis_v2/README.md`与`QUESTIONS.md`。v2将“模式对象不匹配”与“规则改变”区分，增加BASE/S2G原有oracle和同分母费用，原v1仍保留。脚本只分析、不调用模型、不修改预测；输出目录必须未存在。

## 原500题工具

这些脚本只读已有实验记录；不调用模型 API、不读取 API 密钥、不训练或更新记忆、不修改原始 run。它们针对固定的 `500_v1` train-development 实验及其 manifest SHA，不是通用 benchmark 工具，也不把开发集结果称为官方测试复现。

在项目根目录运行，先安装本项目及测试依赖。命令中的输出目录必须是新目录，不能覆盖旧结果。

| 工具 | 输入与用途 | 明确边界 |
| --- | --- | --- |
| `audit_shared500.py` | 已封口批次的账本、journal、API 原始审计及报告 | 未知费用不是零；已知费用是 token 估价，不是供应商账单；不重复累计历史预算 |
| `request_profile_audit.py` | 原始API审计中的模型、角色、容量、JSON schema及请求标识 | 不输出消息/响应正文/凭据；本地一致性不是供应商公证；T0不保证确定输出 |
| `trajectory_stats.py` | 已封口批次的执行轨迹、来源快照和评分报告 | 重复 query 不等于无效；分类不是因果分析；失败与完整配对分开 |
| `missing_bounds.py` | 聚合 `analysis.json` 与固定 manifest | 缺失结果的确定性最坏/最好边界，不是补分、置信区间或显著性检验 |
| `query_replay_probe.py` | 无 gold 执行文件、SQLite 索引、固定作者快照；重放冻结后才读 gold | 相同真实状态下比较支持原文到达；无 Reader 调用，不能推断答案收益；需要额外离线 CPU |
| `export_shared500_costs.py` | 复用API审计，逐题/逐方法/逐角色导出成本 | 必须是全部28批500题3350调用；未执行为null、失败计费、未知不补零，不能冒充任意未来run的通用导出 |

```powershell
.venv/Scripts/python.exe scripts/audits/audit_shared500.py --output runs/shared500_api_audit_final
.venv/Scripts/python.exe scripts/audits/request_profile_audit.py --output runs/shared500_request_profile_final.json
.venv/Scripts/python.exe scripts/audits/trajectory_stats.py --output-dir runs/shared500_trajectory_final
.venv/Scripts/python.exe scripts/audits/missing_bounds.py --analysis runs/CHOOSE_AGGREGATE/analysis.json --manifest data/hotpotqa/shared500_sep27_v1/manifest.json
.venv/Scripts/python.exe scripts/audits/query_replay_probe.py --all-closed --workers 2 --output runs/shared500_query_replay_final
.venv/Scripts/python.exe scripts/audits/export_shared500_costs.py --output runs/shared500_cost_export_new
```

`audit_shared500.py` 和 `trajectory_stats.py` 可以只分析当前已封口部分，必须查看实际分母。`missing_bounds.py` 也可在未完成阶段运行，此时未启动题同样使界限变宽。

`query_replay_probe.py --all-closed` 必须等固定 500 题全部有工程封口记录；它再核对精确题号与 offset 0–499。失败题仍在工程完成记录中，但不进入完整配对比较。小规模验证使用 `--limit 2`（最多 6），默认 1 个进程；`--workers 2` 是允许的最大值。每个进程单独打开只读 SQLite，输出按 offset/round 排序。逻辑检索次数、索引调用、每题缓存命中分别报告，缓存不是在线方法优势。

## 测试与可选本地资源

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_shared500_api_audit.py tests/test_shared500_query_replay.py tests/test_shared500_trajectory_stats.py tests/test_shared500_missing_bounds.py tests/test_shared500_request_profile.py tests/test_shared500_cost_export.py
```

绝大多数测试只用合成数据，不要求题库、API、密钥或索引。query replay 的两项可选集成测试会使用 `external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6`：若本地快照缺失，明确跳过；已有文件的 SHA 不匹配则失败。测试不会自行下载第三方代码。

仓库只包含我们编写的审计逻辑与合成测试；不提交作者源码快照、数据集、API 响应、原始运行记录或凭据。人工整理的结果报告会入库，原始审计留本地。全量 query replay 需要自行准备并核验上述快照和现有只读索引；其余五个工具不执行作者代码。

已生成的逐题成本：`runs/2026-09-27_shared500_cost_export_v1/`。`per_question_costs.jsonl`每行一道题，内含两臂和角色小计；`summary.json`区分全部尝试与492完整配对。查看这些文件不会重新调用API；重复导出必须换新目录。
