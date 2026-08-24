# GrowRAG 当前项目记忆

更新日期：2026-08-24
状态：方案 B「可信自适应查询修复」与三层记忆方向已确认；证据状态、文档层字段、QPP 和双控制器处于讨论确认阶段，尚未修改运行代码
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

## 证据充分性与 FRESH 闭环

1. “证据充分”定义为：原问题的每个必要 information need 都能指向当前累计证据，且没有未解决冲突；不能用“LLM 看起来能回答”替代。
2. 下一版建议使用三态：
   - `SUPPORTED`：所有必要项有证据，允许回答；
   - `REPAIRABLE_GAP`：能显式指出缺的实体/属性/关系/桥接证据，进入修复；
   - `CONFLICTED_OR_UNCERTAIN`：证据冲突、问题含糊或 judge 不稳定，继续核验或弃答。
3. 每轮保存 `needs -> evidence_refs` 支持图、结构化 gaps、contradictions 和 progress；先映射到现有 `sufficient / insufficient / unknown`，确认后再升级 schema。
4. FRESH 是同一 QueryEpisode 内的现场修复：始终以原问题 `q0` 为目标，根据累计证据的当前 gap 生成下一条 query；不调用跨题 ExperienceCard。
5. 默认终止建议：充分即答；总共最多 4 次检索（1 BASE + 最多 3 repair）；连续 2 轮无 gap closure/新证据、query/gap/证据循环、持续冲突或预算耗尽则 `STOP_ABSTAIN`。具体阈值只在 dev 校准。
6. 回答前 coverage gate 是主路由；回答后的 claim support check 暂作可开关安全项，不把两个 judge 混成一个含糊分数。

参考边界：S2G-RAG 已覆盖累计证据上的二值 sufficiency 与结构化 gap；ReflectiveRAG 已覆盖本题内 `Sufficient/Refine`；Self-RAG/SURE-RAG 可参考回答后的支持检查。因此“证据不足后再搜”不是创新。

## 记忆设计

1. `QueryEpisode`：一次问题从原查询到最终回答/弃答的完整具体轨迹；保存实际 query、gap/support map、证据指针、成本和结果，可触及多篇文档。
2. `DocumentSession`：**严格的一文档一版本局部记忆**；保存该文档的 alias/术语、section/chunk landmarks、ambiguity/failure hotspots、`gap × action` 局部 benefit/harm 统计及来源 episode-turn 指针，文档结束后冻结。它不保存答案、完整 episode、文档正文或跨文档规则。
3. `ExperienceCard`：从 verified beneficial episodes 晋升的跨问题、跨文档程序记忆；保存去实体化的 query/gap pattern、前置/禁用条件、repair form/intent、evidence contract、可靠性账本和版本来源。
4. 所有完成的 episode 都冻结；错误/未知 episode 进入隔离审计档案。正确性决定能否晋升，不决定是否保存。
5. v1 在线只调用成功且经独立验证有增益的卡；失败 episode 不进入 serving memory。
6. 采用具体—局部—抽象表示：具体 `q -> q'` 永久留在 QueryEpisode；单文档导航知识放在精确版本的 DocumentSession；长期卡保存去实体化 operator/条件/证据契约，并指回来源 episode/turn。
7. 不保存完整 CoT、历史答案或文档正文到 active card；保存结构化 gap、动作、证据指针、成本和可验证结果。
8. 卡片合并或修订生成新版本，不覆盖来源或旧卡。
9. `serving memory` 统一称“在线可用库”：只有 active cards 和精确版本匹配的文档层可影响当前 query；`cold archive` 统一称“审计档案库”：保存冻结 episode、失败、隔离/退役卡和旧版本，默认不进入在线 prompt。
10. `forget/retire` 默认只从在线可用库移除，不物理删除原始记录；合并生成带 parent 的新版本。

三层的一行定义：QueryEpisode 记录“这道题发生了什么”；DocumentSession 记录“在这一篇文档里怎样找”；ExperienceCard 记录“跨文档仍可能怎样修”。三层本身不是创新，Useful Memories 与 SegMem-RAG 已明显覆盖相邻思想。

## 成功与晋升

- `working_success`：修复后答对，但没有 DIRECT 对照；
- `beneficial_success`：同环境/预算下优于 BASE/DIRECT，或补回其缺失的 gold supporting facts；
- `trusted_success`：在独立 query/document 上重复有益且 conditional harm 低。

单次成功只能成为 exemplar/candidate；第 2 层才可生成 candidate card，第 3 层才可 active。

验证来源优先级：gold / 人工 > 多 judge 一致 > proxy。只有 Recall、相似度、QPP 或 sufficiency proxy 不能直接证明 trusted。

## REUSE 选择：可靠性与适用性分离

- `reliability` 回答“这张卡过去在独立题/文档上是否相对 BASE 反复有益、伤害率是否低”，是卡片级、慢更新历史统计；
- `applicability` 回答“可靠的卡是否适合眼前的 q0、gap、文档版本与 retriever 能力”，是每题重算的瞬时匹配。

选择顺序固定为：reliability 硬门 -> applicability 前置/禁用/能力约束 -> query/gap/action 结构匹配与 QPP 排序 -> 执行后的 evidence contract 验证。高适用性不能挽救低可靠经验，历史可靠也不能挽救当前不相关经验。

RRM 已保存 applicability conditions、required evidence、query-adjustment patterns，并做衰减、合并和裁剪；这些字段与 top-k 淘汰不能归我们。GrowRAG 的候选增量是双轴显式分离、同题 paired harm 和执行后履约验证。

## 查询变换与 QPP

首版不发明新 query rewriter，把已有方法作为可选 operator/基线：

- LLM rewrite / Rewrite-Retrieve-Read；
- Query2doc expansion；
- HyDE（主要用于 dense retrieval）；
- decomposition；
- disambiguation、add constraint、bridge entity、evidence focus。

接口不采用一个混杂的扁平枚举，而分成：

- `QueryDecision = KEEP | TRANSFORM`；
- `transform_form = PARAPHRASE | EXPAND | DECOMPOSE | HYDE`；
- `repair_intent = DISAMBIGUATE | ADD_CONSTRAINT | BRIDGE_ENTITY | FILL_ATTRIBUTE | FILL_RELATION | EVIDENCE_FOCUS`。

LLM rewrite 描述生成手段；expansion/paraphrase/decomposition 描述变换形式；HyDE 更准确地说是检索表示变换。QPP 只作为当前候选的低成本预期检索质量特征，不判断 sufficiency、reliability 或最终答案正确性。v1 优先线上 pre-retrieval QPP，post-QPP 用作离线对照/分析。

## Prompt 与双控制器

v1 冻结三个 prompt：`state_judge`、`fresh_repair`、`experience_apply`。

在线 `RetrievalController` 观察 EvidenceState、预算、QPP 和候选卡的 reliability/applicability，选择 `ANSWER / REUSE_TRANSFORM / FRESH_TRANSFORM / STOP_ABSTAIN`。它不只是两个失败分支。v1 用冻结规则/prompt。

异步 `MemoryLifecycleController` 在回答后选择 `NOOP / CREATE_CANDIDATE / PROMOTE / MERGE_AS_NEW_VERSION / QUARANTINE / RETIRE_FROM_SERVING`。QueryEpisode 每题结束自动冻结，不由 controller 决定是否保存。该分离参考但不同于 MemCon 把在线调用和 Consolidate/Forget/NoOp 放入同一 MDP。

以后训练的小 controller 只选择动作，不负责生成改写。训练标签来自 train/dev 上同题实际执行 BASE/REUSE/FRESH 后的 full-information oracle：选择达到质量阈值、成本最低且不产生 correct-to-wrong harm 的动作。先从规则、逻辑回归或树模型开始，不从 RL/bandit 开始。

## 环境处理

- 每个 episode 只保存 `run_context_id`，实际语料/切块/索引/retriever/reranker/LLM/prompt/judge 版本存在单独不可变注册表。
- v1 固定一个 BASE 环境，严格同环境复用。
- 长期卡只声明必要能力，例如 dense search 或 graph neighbor expansion；跨 retriever/corpus 是后续泛化实验，不要求首版同时解决所有变量。
- EMA/Kalman-inspired gain 只作为后续 priority/recent reliability 更新，不证明当前 applicability，也不是主创新。

## 当前最可守的中心贡献

中心命题：**同题配对、伤害感知的查询修复安全迁移**，而不是“三层记忆 + QPP + actions”。

1. BASE/REUSE/FRESH 在同题、同环境和同预算下配对，不能仅因 treatment 答对就给经验记功；
2. 历史 reliability 与当前 applicability 明确分离，任一不合格都拒绝复用；
3. 结构化 gap 与 evidence contract 让“预测适用”接受执行后核验；
4. 显式统计并约束 `P(REUSE bad | BASE good)`，允许安全退回 FRESH 或普通 RAG；
5. 不可变 episode、精确文档版本与版本化卡片为上述统计提供可审计来源。

三层记忆、QPP、动作词表、衰减/top-k/merge/forget 和 sufficiency loop 都是支撑模块或基线，不能并列包装成多个“创新”。这仍可能被审稿人评价为 RRM + S2G + ReFormeR + Useful Memories 的组合，必须用 paired-benefit、reliability/applicability、harm gate、scope-aware memory 等消融回答。

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
- SIM-RAG：候选答案与累计证据的 Accept/Reject critic；不等同于回答前 coverage；
- How Memory Management Impacts LLM Agents：经验跟随可能传播错误，支持用未来独立任务验证记忆质量。

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

1. 等用户确认 2026-08-24 架构讨论稿；
2. 将现有 DocumentSession v0 归组容器升级为严格文档级 Schema v1，并补 EvidenceState/两轴 QueryTransform 合同；
3. 固定一个文本 BASE RAG，跑 BASE/FRESH paired trajectories；
4. 冻结 state/gap 与 repair prompts；
5. verified beneficial episode → candidate card；
6. 接入 REUSE gate，完成三臂 oracle；
7. 验证跨 query/document 的 benefit/harm/coverage/cost；
8. 现象成立后训练小 controller；
9. 最后换另一种 BASE 检验 sidecar 兼容。

## 代码与仓库状态

- 现有 `src/growrag` 是路线 A 的可信复用核心，可作为方案 B 的 REUSE 子模块，不再代表完整系统。
- QueryEpisode、结构化 state、配对 VerificationEvent、ExperienceCard 和可信 Registry 的 v0 数据合同已实现；当前 DocumentSession 仍只是 v0 归组容器，真实 judge/QPP/controller/retriever 尚未接入。
- 2026-08-24 的三层记忆与 EvidenceState 是待确认 Schema v1；本轮只更新文档，没有提前改代码。
- 本地 `origin` 已指向 `https://github.com/Keep-Passionate/growRag.git`；2026-08-22 已完成首次发布，此后继续同步 `main`。
- GitHub 连接器不是 Git 提交/推送的必要条件；当前本机 Git 凭据已能完成仓库同步。

## 更新规则

- 方向改变先更新 `knowledge/decisions/`，再同步本文件；
- 任何新论文若覆盖 provenance、paired harm 或 applicability，立即重做 novelty matrix；
- 所有 episode 完成即冻结，验证只追加；
- 每次实验先写假设、预算、公平对照与停止条件；
- 不使用“首次”措辞，直到投稿前完成系统检索与引用追踪。

## 当前详细讨论入口

- `knowledge/method/2026-08-24_证据充分性_三层记忆_QPP与双控制器_讨论稿.md`
