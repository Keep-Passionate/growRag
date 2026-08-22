# GrowRAG 当前项目记忆

更新日期：2026-08-22
状态：方案 B「可信自适应查询修复」已确认；首版采用 BASE-first、冻结 prompt 和 success-only serving memory
权威性：当前入口；旧路线 A 决策保留为历史记录，但不再代表当前主线

## 当前主问题

> 在不修改基础 RAG 的条件下，能否把经过验证的单问题查询修复轨迹，以保留来源的方式晋升为长期经验，并通过历史可靠度、当前适用性和 BASE→REUSE 配对伤害来安全决定是否调用？

暂定简单标题：

- 中文：检索增强生成中的跨任务查询修复可信复用
- 英文：Safe Cross-Task Reuse of Query Repairs in RAG

## 已确认的架构

1. 基础 RAG 是可替换但在一次实验中冻结的 `BASE`，可以是普通向量/混合 RAG，之后再验证 GraphRAG；经验层不修改其索引、retriever 或 generator。
2. v1 采用 `BASE-first`：原查询先检索一次；证据足够就直接回答，不进入复杂路径。
3. 首检不足时输出结构化 gap，再选择：
   - `REUSE_REPAIR`：有历史上可靠且当前适用的经验；
   - `FRESH_REPAIR`：没有安全可用的历史经验；
   - `STOP_ABSTAIN`：无进展或预算耗尽。
4. 检索前 `PROACTIVE_REUSE` 只作为后续消融，不是首版默认。
5. 原路线 A 的 reliability/applicability/harm gate 被吸收到方案 B，作为 REUSE 的安全部件；路线 A 不再单独作为论文。
6. 原方案 C 不做主线；充分、无进展和预算停止只是方案 B 的必要控制。

## Agentic / Adaptive 定位

- 标准术语用 **Agentic RAG**。
- 规则和冻结 prompt 阶段：带 Agentic 修复循环的 adaptive/modular RAG。
- 控制器能够按状态选择 `ANSWER / REUSE / FRESH / STOP` 后：轻量级单控制器 Agentic RAG。
- 多 Agent、插件式、不修改底座、按简单/复杂路由都不是创新。
- 路由目标不是表面“问题简单不简单”，而是“离开当前 BASE 是否值得，是否可能造成负迁移”。

## 记忆设计

1. `QueryEpisode`：一次问题从原查询到最终回答/弃答的完整局部轨迹，是核心最小单位，不等于单文档。
2. `DocumentSession`：可选容器，把同一文档/阶段的多个 episode 归组，结束后冻结；不维护第二份会覆盖的总结。
3. `ExperienceCard`：从 verified beneficial episodes 晋升的跨问题程序记忆。
4. 所有完成的 episode 都冻结；错误/未知 episode 进入隔离审计档案。正确性决定能否晋升，不决定是否保存。
5. v1 在线只调用成功且经独立验证有增益的卡；失败 episode 不进入 serving memory。
6. 采用具体—抽象双表示：具体 `q -> q'` 永久留在 episode；长期卡保存去实体化 operator/条件/证据契约，并指回来源 episode/turn。
7. 不保存完整 CoT、历史答案或文档正文到 active card；保存结构化 gap、动作、证据指针、成本和可验证结果。
8. 卡片合并或修订生成新版本，不覆盖来源或旧卡。

## 成功与晋升

- `working_success`：修复后答对，但没有 DIRECT 对照；
- `beneficial_success`：同环境/预算下优于 BASE/DIRECT，或补回其缺失的 gold supporting facts；
- `trusted_success`：在独立 query/document 上重复有益且 conditional harm 低。

单次成功只能成为 exemplar/candidate；第 2 层才可生成 candidate card，第 3 层才可 active。

验证来源优先级：gold / 人工 > 多 judge 一致 > proxy。只有 Recall、相似度、QPP 或 sufficiency proxy 不能直接证明 trusted。

## 查询修复算子

首版不发明新 query rewriter，把已有方法作为可选 operator/基线：

- LLM rewrite / Rewrite-Retrieve-Read；
- Query2doc expansion；
- HyDE（主要用于 dense retrieval）；
- decomposition；
- disambiguation、add constraint、bridge entity、evidence focus。

“Query Transformer”暂按 query transformation 的接口总称理解；若指特定论文需提供题名。

## Prompt 与控制器

v1 冻结三个 prompt：`state_judge`、`fresh_repair`、`experience_apply`。

以后训练的小 controller 只选择动作，不负责生成改写。训练标签来自 train/dev 上同题实际执行 BASE/REUSE/FRESH 后的 full-information oracle：选择达到质量阈值、成本最低且不产生 correct-to-wrong harm 的动作。先从规则、逻辑回归或树模型开始，不从 RL/bandit 开始。

## 环境处理

- 每个 episode 只保存 `run_context_id`，实际语料/切块/索引/retriever/reranker/LLM/prompt/judge 版本存在单独不可变注册表。
- v1 固定一个 BASE 环境，严格同环境复用。
- 长期卡只声明必要能力，例如 dense search 或 graph neighbor expansion；跨 retriever/corpus 是后续泛化实验，不要求首版同时解决所有变量。
- EMA/Kalman-inspired gain 只作为后续 priority/recent reliability 更新，不证明当前 applicability，也不是主创新。

## 当前最可守的贡献候选

1. 不可变单-query repair episode；
2. 具体 q→q' 与抽象 operator 双表示；
3. 保留来源、版本化、可回滚的晋升；
4. 区分“答对”“相对 BASE 有益”“跨题可信”；
5. 历史 reliability 与当前 applicability 分开；
6. 显式统计 `P(REUSE bad | BASE good)`，允许拒绝历史经验；
7. BASE/REUSE/FRESH 的相同预算配对协议与 risk–coverage/benefit/cost 评价。

这仍可能被审稿人评价为 RRM + S2G/SPARKLE + Useful Memories 的组合，必须通过 raw/abstract/dual、覆盖/版本化、reliability/applicability、harm gate 等消融证明整体设计的必要性。

## 已确认不能作为创新的内容

- Adaptive-RAG 式简单/复杂路由；
- 小模型、bandit 或 RL 控制检索；
- plug-and-play / frozen base / Agentic RAG；
- 显式 gap 后生成 query；
- query rewriting、HyDE、Query2doc、decomposition；
- episodic + consolidated 双记忆；
- applicability conditions、query-only reuse、成功/失败库、生命周期衰减；
- 经验驱动地选择 retriever/strategy。

## 关键近邻

- Adaptive-RAG：复杂度路由；
- C-3PO / SPARKLE：冻结底座上的 plug-and-play agentic controller；
- S2G-RAG / Skill-RAG：结构化失败/gap 到定向修复；
- RRM：跨任务程序检索经验、applicability、成功/失败库与 query-only reuse；
- ReFormeR：历史 query-reformulation pattern 的抽取与选择；
- MemCon：旁路 memory controller、NoOp 与在线记忆操作；
- ReMe：正负经验提炼、适应性复用与 utility 生命周期；
- SegMem-RAG：episodic/procedural/semantic memory 与经验路由；
- Useful Memories：持续 consolidation 会损坏记忆；
- GAM-RAG：sentence-level gain-adaptive retrieval memory；
- Experience-RAG Skill：场景 + 经验 + 检索策略路由的可插拔层；
- ERSkill：记忆检索技能、query router、experience trie 与 capability/deploy 双前沿；
- QPP Query Variant Selection：检索前/后选择 query variant。

完整矩阵见 `knowledge/literature/2026-08-22_方案B自适应路由与新颖性边界.md`。

## 数据与实验原则

沿用已确定的 gold 路线，并按方案 B 补充 turn-level 轨迹：

1. 小样本 TREC-RAG/QPP 资源：校验 query variants、retrieval metric 与 answer metric；
2. HotpotQA fullwiki + 2WikiMultiHopQA：主跨题、跨文档 repair transfer 与 harmful reuse；
3. MuSiQue：困难 gap、无进展停止和拒绝压力；
4. BEIR/BRIGHT/TREC DL：query operator 和跨检索器外部有效性；
5. RAGRouter-Bench：后期跨 Naive/Graph/Hybrid/Iterative BASE 验证，不是首个实现目标。

测试集不建库、不晋升、不选阈值、不调 prompt。所有动作必须在相同检索次数/top-k/context/token 预算下公平配对。

## 最近工作顺序

1. schema 与只追加 episode store；
2. 固定一个文本 BASE RAG，跑 BASE/FRESH paired trajectories；
3. 冻结 state/gap 与 repair prompts；
4. verified beneficial episode → candidate card；
5. 接入 REUSE gate，完成三臂 oracle；
6. 验证跨 query/document 的 benefit/harm/coverage/cost；
7. 现象成立后训练小 controller；
8. 最后换另一种 BASE 检验 sidecar 兼容。

## 代码与仓库状态

- 现有 `src/growrag` 是路线 A 的可信复用核心，可作为方案 B 的 REUSE 子模块，不再代表完整系统。
- QueryEpisode、结构化 state、配对 VerificationEvent、ExperienceCard 和可信 Registry 的 v0 数据合同已实现；下一阶段接入 base-adapter/state judge/controller，不需要推倒已测试的 reliability/harm gate。
- 本地 `origin` 已指向 `https://github.com/Keep-Passionate/growRag.git`；远端仓库已核实公开且为空。
- 当前命令行到 GitHub 443 连接失败；GitHub 连接器安装请求仍待用户授权确认，因此尚未推送。

## 更新规则

- 方向改变先更新 `knowledge/decisions/`，再同步本文件；
- 任何新论文若覆盖 provenance、paired harm 或 applicability，立即重做 novelty matrix；
- 所有 episode 完成即冻结，验证只追加；
- 每次实验先写假设、预算、公平对照与停止条件；
- 不使用“首次”措辞，直到投稿前完成系统检索与引用追踪。
