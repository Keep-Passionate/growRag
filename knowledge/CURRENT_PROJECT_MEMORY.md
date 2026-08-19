# GrowRAG 当前项目记忆

更新日期：2026-08-19  
状态：方向讨论与 Gate 0 / Gate 1 准备中  
权威性：当前入口；后续经双方确认后更新

## 已确认

1. 用户所说的“EMR”按 **ERM（Evolving Retrieval Memory）** 理解。
2. 旧 GrowRAG 代码与旧 A1、M1/M2、v2、v2.1 不再约束新项目；它们已经归档。
3. 第一阶段优先研究检索前适配，但保留检索后修复作为失败后的转向或第二阶段。
4. 不能把“记住成功改写”“检索前选择 query variant”“EMA”“经验可选”单独作为创新。
5. 最值得继续核验的切口是：
   - 历史可信度与当前适用性的分别估计；
   - 对 `DIRECT 正确 → REUSE 错误` 的显式风险控制；
   - 不确定时拒绝复用并保持原查询。
6. EMA 只作为后续生命周期与近期可靠度的可选平滑模块，不是主贡献，也不证明当前适用性。
7. 第一版不从强化学习或 bandit 开始；先用有 gold 的数据做完整反事实测量。
8. 数据路线暂定：
   - TREC-RAG 做小规模 oracle 与多阶段指标校验；
   - HotpotQA fullwiki + 2WikiMultiHopQA 做核心跨题迁移和 harmful reuse；
   - MuSiQue 做困难、充分性与拒绝压力测试；
   - BRIGHT/BEIR/TREC DL 负责 ERM 复现与跨检索器外部有效性。

## 当前候选主线

暂称 **Safe-ERM + Do-No-Harm Query Reuse**：

1. 继承 ERM 的“正确性门、原子归因、稳定积累”思想；
2. 将 ERM 的绝对正确门升级为相对 `DIRECT` 的成对改进门；
3. 成功一次只进入候选区，不立即视为 trusted；
4. 用独立目标题上的迁移记录建立历史可靠度；
5. 对当前问题单独估计 applicability；
6. 两者都满足阈值才 `REUSE`，否则 `DIRECT`；
7. 主评价显式报告 benefit、harm、coverage、成本和 risk–coverage 曲线。

这仍是候选路线，不是已经证明的新颖贡献。

## 尚未决定

- 主论文是“安全查询经验复用”还是“可回滚条件化 ERM 索引”。
- 第一版经验表示是具体 query pair、原子 expansion units，还是 ReFormeR 风格 pattern。
- 只做 `DIRECT/REUSE`，还是把 `FRESH` 也纳入在线控制器。
- 风险控制先用经验阈值/校准器，还是进一步追求 conformal 风险保证。
- 阶段 0 是先复现 ERM 还是先跑 TREC-RAG/Hotpot 风险矩阵；两者都保留，默认优先风险矩阵、并行做轻量 ERM 复现。

## 术语约定

- `DIRECT`：当前原查询，不用历史变换。
- `REUSE`：把一条历史查询变换应用到当前问题。
- `FRESH`：只为当前问题现场生成新变换。
- 历史可信度（reliability/trust）：过去独立复用时的收益与伤害记录。
- 当前适用性（applicability/fit）：在当前题执行前，对“这条经验是否匹配”的预测。
- 成对伤害（harm）：`DIRECT` 达标而 `REUSE` 不达标。
- 成对收益（benefit）：`DIRECT` 不达标而 `REUSE` 达标。
- Gold：数据集提供的参考答案、相关文档、supporting facts、qrels 或 nuggets；不是模型自评分。
- Oracle：离线看过所有候选真实结果后的上界，不是可部署路由器。

## 更新规则

- 每次双方确定一项方向，先更新 `decisions/`，再同步本文件。
- 每次实验先写假设与停止条件，再运行。
- 新论文若改变新颖性边界，必须在 `literature/` 记录“做了什么 / 没做什么”。
- 测试集不参与阈值选择、经验建库或提示词调整。
