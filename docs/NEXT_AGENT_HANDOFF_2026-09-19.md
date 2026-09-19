# 下一位执行模型：GrowRAG 2026-09-19交接

## 你现在只需完成什么

先完成下面TASK-01～03的离线代码与测试，每个任务单独提交，完成一项先报告。**不要立即跑付费API、全量Hotpot、训练选择器或重构整仓。** 付费阶段按主计划在账本和协议验收后另启动。

工作目录：D:/project/GrowRAG。
已审代码基准：822a2e0b7e217ba93df8e1cefdb68b2594549b06。先git status识别后续修改，不回滚用户文件。
规划模型建议：Sol medium做一般接线；跨模块/账本用Sol high，遇到研究含义不清时回报而非自由改题。

## 先读这四处

1. knowledge/CURRENT_PROJECT_MEMORY.md 顶部最新记录。
2. knowledge/decisions/2026-09-19_两层记忆定稿与可检验贡献.md。
3. knowledge/experiments/2026-09-19_最小实验与低成本模型交接.md，尤其E0/E1与数据预算边界。
4. src/growrag/outer_loop.py、experiments/paired_runner.py、episodes/models.py、experience/cards.py及对应测试。

## 已定，不再重新设计

- 离线学习/在线应用；最多两次检索，不强制每题二搜。最终PRE入口保留，最小POST实验先固定PRE=BASE。
- 本题局部状态＋跨题程序性卡；DocumentSession后置。
- BASE/FRESH/REUSE是动作来源，不是三层。
- gold只在隔离评分阶段使用；主评价冻结长期库。
- 主归因是同共同前缀、同预算下REUSE−强FRESH，不只比BASE。
- 候选动作冻结，只改变选择器看的经验视图；不能同时改执行规则。
- 不学习一轮/两轮质量乘数来给自己评分。成本统一效用后置，暂不RL。
- 无论文效果保证；两轮、更多记忆、更长条件均可能更差。

## TASK-01：POST共享快照与累计证据

目标：同一BASE前缀可独立分支FRESH/REUSE；第二轮Reader能读到限长后的两轮证据。
优先借用PairedRunner已有合并与原q Reader逻辑，用一个小适配器接outer_loop；不要另起一个大框架。

验收：原q不变；同ID不同文本拒绝；分支证据不串；去重/裁剪确定；引用只指向实际Reader上下文；无证据能力与空检索分开；同一前缀实际采集费计一次；最多两次检索/重复与错误停。

特别注意RagReply目前把evidence解释为本次回答可引用集合。若改为累计Reader，必须明确区分“本轮新检索证据”和“本次Reader使用证据”，不能让增量统计把累计证据又算一遍。用显式小字段或适配层，保留旧合同。

## TASK-02：结构化状态与比较协议

目标：新增最小的requirements/support-links/gaps及比较记录。
judge是冻结prompt调用的有误差工具；仅验证证据ID可追溯不等于判断正确。先用合成固定回复测试，标mock。

比较记录明确reference_strategy、treatment_strategy、共同prefix hash、阶段、预算、方法/模型/prompt版本及反馈来源。
旧TransferObservation的direct_score不接收FRESH分数；旧FrozenQueryEpisode强制BASE-first，修复还必须绑定真实prior gap。无明确gap的语义修复用新实验记录，不伪造gap或PRE-first存档。新POST候选与PRE候选隔离，旧6卡不动。

验收：换gold不影响运行决策；不同前缀拒绝配对；未知不变充分；已充分早停；源题同文本换ID排除；错误保留unknown；旧schema仍可读取。此阶段不接自动ACTIVE晋升、不写测试gold到卡。

## TASK-03：安全续跑和报告合同（仍零API）

目标：完成跨进程预算承接、明确新入口dry-run与逐题报告。
旧账本：
runs/2026-09-14_fresh_baseline_32_v1/final_budget.json
runs/2026-09-14_fresh_baseline_32_v1/cumulative_budget.json

累计保守预留0.3199766元，1次失败费未知；总上限5元不重开。账本不能仅信交接文字，读取原账本核验。失败不自动重试，已成功请求不重放；不删旧锁。

验收：请求前/响应后落盘前/落盘后崩溃的恢复；未知费用保留；配置变化不误用缓存；密钥/错误正文不进日志；mock/real不混；聚合与组件费不双计。新runner入口未存在前不要写假命令。

报告必须可读到：原题→首query/证据→状态/gap→候选卡及版本→选择原因→实际改写→新增与累计证据→答案→事后EM/F1/支持→费用/终止原因。选择依据不能用事后gold包装。

## 数据和作用域红线

- 原来源64/6张PRE卡、原8/24表示目标、新32FRESH开发名单不改、不换名成盲测。
- POST新source/target manifest另建，未验收前不查看新目标gold。
- 当前句BM25/distractor不是fullwiki；不声称复现论文榜单。
- 不安装/微调7B、72B，不租卡，不读出或打印qwenAPI.md。
- 不复制未核许可证的ReFormeR代码；同思想清洁适配与原版复现分开。
- 先不开EMA/自动合并/在线长期写回；不为凑卡降低录入标准。

## 已存在的本地验证命令

运行相关单测后，再运行：
.venv/Scripts/python.exe -m pytest -o addopts='' -q
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
git diff --check

最后一次完整本地测试1057通过发生于09-16，不是你本轮已通过。实际重跑后报告新数量与结果。

## 每项交付格式

- 做了什么，改了哪些模块，哪些刻意没做。
- 哪些验收测试通过/没通过；真实命令和状态。
- 本轮真实API次数必须0；模型质量没有新结论。
- 当前Git提交，未完成项，下一项入口。
- 若测试发现设计冲突，保留失败用例解释，不改测试定义掩盖问题。

本交接不是后台任务，当前没有自动付费执行。计划新增字段/适配器还不是已完成代码。
