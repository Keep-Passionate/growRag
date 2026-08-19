# GrowRAG 当前项目记忆

更新日期：2026-08-19
状态：路线 A 已确认；v1.1 已加入结构化经验卡、热记忆与可调风险门，下一步接入 Gate 1 真实数据
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
9. 2026-08-19 用户正式选择 **查询侧可信复用层**，不再把可回滚 ERM 索引作为首篇主线。
10. v1 默认保存具体 `q → q'`、原子变化和 provenance；在线动作固定为 `DIRECT/REUSE`，`FRESH` 仅作离线强基线。
11. v1 先使用经验阈值、条件破坏率的 Wilson 上界和 risk–coverage；EMA、conformal、bandit/RL 均后置。
12. v1 已实现经验账本、候选召回、适用性接口、可信 gate、target query plan 接口和离线 candidate-set oracle；真实检索器与数据尚未接入。
13. v1 已提供严格 TOML 配置、可重启 JSON 经验快照、请求/计划文件协议、完整 TrustedReuseLayer、统一 `growrag` CLI 和文件化 DIRECT/REUSE 示例。
14. Query variant 指同一信息需求送给检索器的不同表达；来源题 `q_s'`、历史变换卡和应用后的目标题 `q_t'` 必须分开。
15. 经验卡采用 S2G 启发的结构化缺口元数据：来源失败诊断、gap category、修复原子、正向适用条件和禁用条件；它不是当前题看过证据后的 S2G gap item。
16. 生命周期新增非破坏性 hot/cold/expired 视图：保守历史质量乘时间衰减后选 Top-K；衰减只改调用优先级，不能改写可靠度，未验证的调用不刷新年龄。
17. 环境兼容提供 strict/tiered 两种策略；主实验默认 strict，tiered 只有在开发集校准后启用。
18. 安全力度通过 risk–coverage 操作点选择，不追求一味严格；阈值在 calibration/dev 上选定后冻结。
19. 检索后路线若继续，必须研究“跨题 gap→repair 经验能否安全替代 fresh repair”；单纯充分性判断和结构化 gap 已被 ReflectiveRAG、S2G-RAG 覆盖。
20. Contraindication 是独立硬否决，不能因适用性阈值设为 0 而失效；signature 请求必须记录冻结的 query-only extractor ID/version。
21. risk–coverage 的分母必须是完整目标题集合，包括无候选、环境不兼容和计划失败的题；paired outcome 只能由原始分数自动生成。
22. schema v1 只允许审计/迁移，不能直接参与运行决策；经验卡经人工复核后另存 schema v2。

## 已确认主线

暂称 **查询侧可信复用层**：

1. 继承 ERM 的“正确性门、原子归因、稳定积累”思想；
2. 将 ERM 的绝对正确门升级为相对 `DIRECT` 的成对改进门；
3. 成功一次只进入候选区，不立即视为 trusted；
4. 用独立目标题上的迁移记录建立历史可靠度；
5. 对当前问题单独估计 applicability；
6. 两者都满足阈值才 `REUSE`，否则 `DIRECT`；
7. 主评价显式报告 benefit、harm、coverage、成本和 risk–coverage 曲线。

方向已经确认，但新颖性和有效性仍需 Gate 1 实验验证。

## 已冻结的第一版取舍

- 主论文对象：查询侧安全复用，不修改基础索引。
- 经验表示：具体 query pair + 原子变化；pattern/CoT 笔记后续消融。
- 在线动作：`DIRECT/REUSE`；`FRESH` 只作为离线基线。
- 风险控制：先经验校准，不先承诺 conformal 保证。
- 实验顺序：优先风险矩阵与 oracle；ERM 只在需要比较索引侧写入时作为支线基线复现。

## 术语约定

- `DIRECT`：当前原查询，不用历史变换。
- `REUSE`：把一条历史查询变换应用到当前问题。
- `FRESH`：只为当前问题现场生成新变换。
- Query variant：同一信息需求的另一种实际检索查询，可由改写、扩展、分解或消歧产生。
- 历史可信度（reliability/trust）：过去独立复用时的收益与伤害记录。
- 当前适用性（applicability/fit）：在当前题执行前，对“这条经验是否匹配”的预测。
- 成对伤害（harm）：`DIRECT` 达标而 `REUSE` 不达标。
- 条件破坏率：在 `DIRECT` 原本达标的题中，被 `REUSE` 改到不达标的比例。
- 成对收益（benefit）：`DIRECT` 不达标而 `REUSE` 达标。
- Gold：数据集提供的参考答案、相关文档、supporting facts、qrels 或 nuggets；不是模型自评分。
- Oracle：离线看过所有候选真实结果后的上界，不是可部署路由器。
- Priority：热记忆调用顺序；不是可靠度或正确率。

## 更新规则

- 每次双方确定一项方向，先更新 `decisions/`，再同步本文件。
- 每次实验先写假设与停止条件，再运行。
- 新论文若改变新颖性边界，必须在 `literature/` 记录“做了什么 / 没做什么”。
- 测试集不参与阈值选择、经验建库或提示词调整。
