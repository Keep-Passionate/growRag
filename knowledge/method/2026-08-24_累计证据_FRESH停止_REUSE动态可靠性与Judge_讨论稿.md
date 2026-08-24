# 累计证据、FRESH 停止、REUSE 动态可靠性与 Judge：讨论稿

更新日期：2026-08-24  
状态：方法设计讨论稿；用于冻结方向，**尚未据此修改运行代码**  
承接文档：`2026-08-24_证据充分性_三层记忆_QPP与双控制器_讨论稿.md`

## 0. 本轮已经明确的结论

1. `CONFLICTED_OR_UNCERTAIN` 不能只有“继续搜”一个出口。它至少有两条主分支：
   - 仍有明确可核验缺口、预算尚可且预计能获得新证据：进入本题内 `FRESH_VERIFY_OR_REPAIR`；
   - 已无进展、循环、预算耗尽或冲突不可安全消解：撤销本轮经验影响，并 `STOP_ABSTAIN`；如果撤销后 BASE 证据本来就充分，则退回 BASE 回答。
2. 累计证据不是把所有检索文本拼接成长 prompt，而是“不可变证据账本 + 去重后的当前证据视图 + need-support 图 + 每轮增量 + 可回滚检查点”。
3. `gap` 的核心思想确实来自 S2G-RAG：先结构化说明缺什么，再生成下一条 query。GrowRAG 需要扩展 gap 的状态、来源和履约字段，但不能把结构化 gap 本身当创新。
4. 部分 need 已得到支持时，只修复仍为 `partial / missing / conflicted` 的部分，不重写整道题；所有必要 need 得到支持且冲突已解决时成功停止。
5. 除“证据充分”和“最大轮数”外，必须加入“继续检索已无明显新增信息”的停止。用户记得的直接近邻是 ReflectiveRAG；AIR 提供了更简单、可复现的未覆盖项停止规则。
6. ExperienceCard 的 `reliability` 必须随经验证的使用结果动态更新，但不能只反复覆盖一个浮点数。应保留不可变试验账本，再计算收益下界、伤害上界和近期趋势。
7. v1 不必训练神经模型，但需要有 gold 的 source/calibration 数据来产生 FRESH 轨迹、构建候选卡并校准阈值；测试集不能用于建库或更新卡。
8. 冻结 prompt 的 Judge 是一个“按固定规则评估证据状态的裁判”，不是回答生成器。冻结是为了控制变量、避免测试泄漏并使不同 episode 的统计可比。
9. Self-RAG 的支持判断可作为回答后核验的启发，但不能说“query 被文档支持”。Query 是检索动作；应判断它是否带来了目标 gap 所需的新证据，以及最终 answer claims 是否被证据支持。
10. 论文中心贡献应收敛为：**同题配对、伤害感知的历史查询修复安全迁移**；Query Transformation、三层记忆、QPP、FRESH 循环都是为此服务的组件。

## 1. 完整在线状态机

```text
原问题 q0
  -> BASE(q0)
  -> 更新 EvidenceLedger / ActiveEvidenceView / SupportGraph
  -> EvidenceJudge(q0, current evidence)

SUPPORTED
  -> ANSWER
  -> AnswerSupportCheck（v1 可开关，正式实验建议开启）

REPAIRABLE_GAP
  -> 若存在可靠且适用的卡：REUSE_REPAIR
  -> 否则：FRESH_REPAIR
  -> 检索、抽取、去重、更新证据，再 Judge

CONFLICTED_OR_UNCERTAIN
  -> 若冲突/歧义能表达成可检索 gap，且预计仍有进展：FRESH_VERIFY_OR_REPAIR
  -> 否则若本状态由 REUSE 引入：ROLLBACK_TO_PRE_REUSE
       -> 回滚后充分：ANSWER_BASE
       -> 回滚后仍不足：STOP_ABSTAIN
  -> 否则：STOP_ABSTAIN
```

### 1.1 为什么需要“回滚”而不只是“不再使用经验”

假设 BASE 找到证据 `e1`，REUSE query 又找来与 `e1` 冲突的 `e2`。如果系统只是从下一轮起不再调用该卡，但仍把 `e2` 当成可靠累计证据，那么经验已经污染了后续 Judge 和答案。因此每次 REUSE 前保存一个轻量检查点：

```yaml
BranchCheckpoint:
  checkpoint_id: cp_before_reuse_2
  active_evidence_ids: [e1, e3]
  support_graph_version: sg_1
  originating_card_id: card_17
  originating_card_version: 4
```

回滚不删除原始检索记录，只把 REUSE 分支中新证据标为 `quarantined_for_current_answer`，恢复之前的 active view。它仍留在 QueryEpisode 的审计账本中，用于给卡片记一次伤害/失约事件。

## 2. “累计证据”具体是什么

推荐把一个口语上的“累计证据”拆成四个数据对象，而不是一个越来越长的字符串。

### 2.1 EvidenceLedger：不可变原始账本

它保存每轮实际取得的最小证据单元及来源。新记录只追加，不覆盖。

```yaml
EvidenceUnit:
  evidence_id: ev_0031
  query_episode_id: ep_0042
  turn_id: 2
  branch_id: reuse_card17_v4

  document_id: wiki_Albert_Einstein
  document_version: 2026-01-dump
  unit_id: sentence_18
  content_hash: sha256:...

  retrieved_by_query: "..."
  query_action:
    decision: TRANSFORM
    form: EXPAND
    intent: FILL_ATTRIBUTE
  card_ref: card_17@4        # FRESH/BASE 时为空

  supports_need_ids: [need_2]
  stance: SUPPORTS           # SUPPORTS | CONTRADICTS | UNCLEAR
  extraction_score: 0.91
  retrieval_score: 11.37
  first_seen_turn: 2
  runtime_text: "..."        # 只在本次运行/受控 artifact 中保存
  serving_status: ACTIVE     # ACTIVE | DUPLICATE | QUARANTINED | REJECTED
```

为什么不能只存文档 ID：Judge 与 supporting-fact 指标通常需要句子/片段粒度。为什么不能只存正文：没有 `document_id + version + unit_id + hash` 就无法证明它来自哪里，也无法判断语料变更。

### 2.2 ActiveEvidenceView：给 Judge/回答器看的紧凑视图

EvidenceLedger 可能包含重复、冲突、低相关和被回滚的单元。当前 prompt 只序列化 active view：

- 按 `content_hash` 去重；
- 对高度近似片段只保留一个正文，但保留多个来源指针；
- 优先保留与未解决 gap 直接相关的句子；
- 不放入被回滚或低可信抽取结果；
- 控制每个 need 的证据数量，避免某一 need 挤占全部上下文。

这与 [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/) 的 sentence-level Evidence Context 方向一致。我们的补充是为 paired reuse 增加 branch/card/version/checkpoint 来源，并明确区分原始账本与当前 serving view。

### 2.3 NeedSupportGraph：问题需要与证据之间的支持图

```yaml
NeedState:
  - need_id: need_1
    description: "确定目标实体 A"
    required: true
    status: COVERED          # COVERED | PARTIAL | MISSING | CONFLICTED
    supporting_evidence_ids: [ev_0012]
    contradicting_evidence_ids: []

  - need_id: need_2
    description: "获得 A 的目标属性 P"
    required: true
    status: PARTIAL
    supporting_evidence_ids: [ev_0031]
    contradicting_evidence_ids: [ev_0035]
```

证据充分不是一个无来源的 `0.82`，而是每个必要 need 都能指向具体 evidence IDs。Judge 可以额外给置信度用于校准，但状态与引用是主体。

### 2.4 ProgressDelta：每轮到底新增了什么

```yaml
ProgressDelta:
  turn_id: 2
  new_unique_evidence_count: 2
  newly_covered_need_ids: [need_1]
  newly_partial_need_ids: [need_2]
  gaps_closed: [gap_1]
  gaps_created: [gap_conflict_1]
  contradictions_added: 1
  contradictions_resolved: 0
  duplicate_ratio: 0.67
  candidate_query_novelty: 0.18
```

它不直接等于“收益”，但为停止、卡片履约和后续重新计算阈值保留可观测原始量。

## 3. Gap 是否就是 S2G-RAG 那种结构

是，核心是同一种范式：

```text
证据不足
  -> 不直接说“再搜一次”
  -> 明确缺哪个实体、属性、关系或桥接事实
  -> 用这个缺口生成下一条 query
```

GrowRAG 的建议 schema：

```yaml
GapItem:
  gap_id: gap_0007
  parent_need_id: need_2
  category: ATTRIBUTE       # ENTITY | ATTRIBUTE | RELATION | BRIDGE | DISAMBIGUATION | CONFLICT
  target: "实体 A"
  slot: "出生地"
  known_context: ["A 的职业为 ..."]
  missing_description: "缺少可验证的出生地及来源"
  status: PARTIAL           # PARTIAL | MISSING | CONFLICTED
  supporting_evidence_ids: [ev_0031]
  contradicting_evidence_ids: []
  expected_evidence:
    must_mention: ["实体 A", "出生地关系"]
    acceptable_source_scope: ["current_corpus"]
  priority: REQUIRED
```

与 S2G-RAG 的差别应该是工程和研究接口上的扩展，不是声称发明 gap：

- gap 指向原始 `need_id`，因此能度量是否真的关闭；
- 增加 `PARTIAL / CONFLICTED`，不仅是“缺失”；
- 保存支持/冲突证据指针；
- `expected_evidence` 可成为 ExperienceCard 的执行后 evidence contract；
- 保存由 BASE、REUSE 还是 FRESH 触发，便于配对归因和回滚。

## 4. 部分支持时如何只做“部分改写”

假设问题需要 `need_1 + need_2 + need_3`，当前已有：

- `need_1 = COVERED`
- `need_2 = PARTIAL`
- `need_3 = MISSING`

下一轮 query 不应把整个原问题改写一遍，而应：

1. 保留原问题 `q0` 作为最终目标；
2. 只选 `need_2/need_3` 对应 gap；
3. 将已经验证的桥接实体作为 grounding，而不是把历史答案直接塞进去；
4. 输出本轮 query 与 `target_gap_ids`；
5. 执行后只检查目标 gap 是否产生新支持。

例如：

```yaml
RepairAction:
  source: FRESH
  target_gap_ids: [gap_2, gap_3]
  form: DECOMPOSE
  intent: BRIDGE_ENTITY
  generated_queries:
    - "..."
    - "..."
  preserved_covered_need_ids: [need_1]
```

[AIR（ACL 2020 Main）](https://aclanthology.org/2020.acl-main.414/) 已经提出“围绕当前 justification 尚未覆盖的 query terms 继续改写”；因此“只修未覆盖部分”是重要基线思想，不是我们的创新。

## 5. 完善后的停止协议

### 5.1 成功停止

同时满足：

1. 所有 `required=true` 的 needs 都是 `COVERED`；
2. 没有未解决的关键 `CONFLICTED` need；
3. 每个 covered need 至少指向一个允许范围内的 evidence unit；
4. 正式安全配置下，最终答案的可核验 claims 通过 support check。

### 5.2 无明显新增信息停止

分执行前和执行后两层。

执行前拒绝候选 query：

- 规范化后与历史 query 完全相同；
- 语义高度近似，并且目标 `gap_ids + action intent` 也相同；
- 没有加入新的实体、约束、关系方向或证据范围；
- 同一 gap-action 已连续失败。

执行后判为无进展：

- `new_unique_evidence_count = 0`；
- 没有 need 从 `MISSING→PARTIAL/COVERED` 或 `PARTIAL→COVERED`；
- 没有关闭 gap、减少冲突；
- 新结果的重复率过高。

默认规则建议：连续两轮满足“无 gap closure 且无新独立证据”，立即停止，而不是用一个 LLM 主观分数继续拖延。

### 5.3 文献对应

- [ReflectiveRAG（EACL 2026 Industry）](https://aclanthology.org/2026.eacl-industry.27/) 使用检索集合相似度与置信变化组成的边际改进量，低于阈值时停止。它就是用户记得的最直接论文；但公开描述中的 `Sim/Conf/阈值` 细节不足以让我们原样复现，应把它作为对照和灵感。
- [AIR（ACL 2020 Main）](https://aclanthology.org/2020.acl-main.414/) 在 query terms 已由 justification 覆盖或没有发现新 query terms 时停止，较简单、透明。
- [S2G-RAG（ACL 2026 Long）](https://aclanthology.org/2026.acl-long.1185/) 以 Judge 判充分或固定预算结束，结构化 gap 很强，但 no-progress/cycle 终止还可补充。
- [HALT（2026-08 arXiv 预印本）](https://arxiv.org/abs/2608.02009) 已经将停止表述成 expected hop claims 与累计证据的逐项覆盖；这直接限制了我们的新颖性，need coverage 只能作为基础设施。
- [SIM-RAG（SIGIR 2025）](https://doi.org/10.1145/3726302.3730018) 训练轻量 Critic 做 Accept/Reject，更接近“候选答案是否可接受”，不是显式 need coverage。
- TASR 使用答案重复与校准 logit margin 做训练自由停止；适合成本基线，但“答案稳定”不等于“证据完整”。
- Stop-RAG 把停止建成有限时域 MDP 并训练价值控制器；适合后期 learned-controller 基线，不应成为 v1 前置依赖。

### 5.4 最大轮数如何确定

先用 `1 BASE + 最多 3 repair` 作为开发默认值，而不是论文结论。理由是它便于与 ReflectiveRAG 的三次 reformulation 设置和 S2G-RAG 的多轮设置对齐，也能控制成本。

正式确定方式：

1. 只在 dev/calibration split 上运行 repair budget `B ∈ {1,2,3,4}`；
2. 对每个 B 记录 answer EM/F1、support recall/F1、平均检索次数、无进展轮占比和 harmful reuse；
3. 选择“达到 dev 最佳质量的容许区间内，成本最低”的 B，例如质量不低于最佳值 `δ`；
4. 在 test 前固定一个全局 B，不按 test 数据集单独调；
5. 正文报告 B=3，附录报告 1/2/3/4 敏感性。

可以写成约束选择：

```text
选择最小 B，使：
  Quality(B) >= max_B Quality(B) - δ
  且 ConditionalHarm(B) <= ε
```

`δ` 与 `ε` 也只能由 dev 及应用风险要求确定，不能看 test 后调。

## 6. ExperienceCard reliability 如何动态更新

### 6.1 不要只保存一个会被覆盖的 confidence

每张卡保存不可变的验证事件：

```yaml
ReliabilityLedger:
  independent_query_count: 12
  independent_document_count: 9
  paired_trials:
    beneficial: 5       # BASE 错，REUSE 对
    harmful: 1          # BASE 对，REUSE 错
    neutral_good: 4     # 两者都对
    neutral_bad: 2      # 两者都错
  contract_fulfilled: 8
  contract_violated: 4
  verifier_tiers:
    gold_or_human: 10
    multi_judge: 2
    proxy_only: 0
  last_validated_at: "..."
  recent_gain_ema: 0.18
```

单次使用后追加事件，再重算 serving score；不让 LLM 把旧统计总结成一句新的“这张卡很可靠”。

### 6.2 建议的双风险统计

把“带来收益”和“造成伤害”分开估计：

- `BenefitPosterior`：在可救援题上带来 `BASE wrong → REUSE correct` 的概率；
- `HarmPosterior`：在 BASE 本来正确时造成 `BASE correct → REUSE wrong` 的概率。

数据少时不要看均值。v1 优先使用易解释的 Wilson 区间：

- repair/benefit 的 Wilson 保守下界 `Benefit_LCB`；
- conditional breakage/harm 的 Wilson 保守上界 `Harm_UCB`。

在线可靠度可以概念性表示为：

```text
reliability_serving = Benefit_LCB × (1 - Harm_UCB)
```

还必须有最少独立 query/document 数。十个近重复问题不能当十次独立证明。

以后若需要概率排序或 Thompson sampling，可把 repair 与 harm 分别建 Beta 后验；不能只建一个“总成功率”。Wilson/Beta 都是原始账本的派生视图，不改变历史事件。

### 6.3 EMA 放在哪里

EMA 只用于近期趋势和召回优先级：

```text
recent_gain_t = (1 - α) × recent_gain_(t-1) + α × observed_gain_t
```

如果最近表现突然下降，可让卡从 `ACTIVE → QUARANTINE`，等待重新验证。EMA 不替代完整账本，也不能单独证明当前适用性。类似 GAM-RAG 的 gain-adaptive 更新和 RRM 的时间衰减可作为参考；我们的重点是将它们限制在“动态优先级/漂移警报”，而不是当作真值。

### 6.4 Top-k、淘汰与冷归档

- 在线 active cards 可以按可靠度、近期使用价值和冗余度维持 top-k；
- 低可靠、过时或重复卡从在线库移入隔离/退役状态；
- 不物理删除其 episode、卡版本和失败事件；
- merge 生成新版本并保留 parents，不能覆盖旧卡。

这能控制在线复杂度，也保留错误分析和未来再验证能力。

推荐生命周期：`CANDIDATE → SHADOW/CALIBRATING → ACTIVE → QUARANTINE → RETIRED`。source 上一次收益只能创建 candidate；必须在独立 query/document 上 shadow 验证后才 active。一次高可信、严重的 `BASE good→REUSE wrong` 可以立即隔离；连续指标上的轻微负增益则由区间与 EMA 累积判断，避免安全门过度敏感。

## 7. 系统是否需要训练数据

### 7.1 简短答案

需要**用于积累和校准的有 gold 数据**，但 v1 不一定需要梯度训练。

空记忆启动时没有 REUSE 卡，系统仍能运行：

```text
BASE -> 不足时 FRESH -> gold/人工验证 -> candidate episode/card
```

经过 source 数据上的离线轨迹后，才有足够候选卡进入 REUSE。所谓“先冻结 prompt”并不代表从零就凭空拥有历史经验。

### 7.2 推荐三段式协议

1. **Source / memory-build**：运行 BASE 与 FRESH；gold 只在执行后揭示；从 verified beneficial episodes 生成 candidate cards。
2. **Shadow / route**：在独立 query/document 上试用候选卡，不让它影响正式输出；积累迁移 benefit/harm，未来也可训练小 router。
3. **Calibration**：校准 reliability、applicability、Judge 和停止阈值；决定 active/quarantine。
4. **Held-out test**：冻结 prompt、模型、阈值、BASE、卡片版本和在线 active set；不创建、合并、晋升或更新卡。

若要研究在线自进化，必须另做 chronological/prequential 实验，按时间顺序在每题结束后更新，且绝不使用未来样本；不能与冻结测试结果混在一起。

### 7.3 数据集阶段

- Pilot：HotpotQA distractor 约 1,000 source、200 calibration、500 held-out，仅用于跑通，不发表主结论。
- 主实验：HotpotQA fullwiki 官方 train 建库、train 内固定一部分做 calibration、官方 dev 做 held-out；2WikiMultiHopQA 同样使用 train→dev，并增加实体/关系模板去重的 strict-transfer 切分。
- 困难/停止压力：MuSiQue 的 2–4 hop、分解与不可回答样本。
- TREC-RAG 的 56 topics 只校验 query variants、QPP 与 oracle，不训练门控。

正式主协议先采用官方 train 内按 group 切分的 `S_exp 40% / S_shadow-route 35% / S_cal 25%`，官方 dev 作为 held-out。这里的 group 必须同时考虑 supporting-document cluster、模板/组成单跳 lineage、答案实体和近重复 query，不能随机逐题切分。pilot 后可以在不看官方 dev 的前提下依据卡片覆盖率调整一次比例，并记录为新协议版本。

### 7.4 以后训练小模型时用什么标签

对 train/calibration 的同一道题，实际执行：

```text
BASE | 每个合格 REUSE 候选 | FRESH
```

在相同预算下用 gold 构造 full-information oracle，标签为 `ANSWER_BASE / REUSE(card) / FRESH / STOP` 中达到质量约束、成本最低且不产生 harmful transfer 的动作。首个学习器建议逻辑回归、树模型或小型 cross-encoder；先不要上 RL/bandit。

## 8. REUSE 四级选择：具体是什么、分别参考谁

### 第 1 级：历史可靠性硬门

问题：这张卡过去是否真的反复比 BASE 有益，并且少造成伤害？

检查：

- verified paired trials 数；
- independent query/document 数；
- benefit 下界和 harm 上界；
- evidence contract 履约率；
- active/quarantine/retired 状态；
- 来源与版本是否完整。

参考：

- RRM：复用反馈、频次、时间衰减、合并与裁剪；
- ReMe：成功/失败程序经验与 utility 生命周期；
- GAM-RAG：收益与不确定性/近期信息的更新思路；
- Useful Memories：不要通过反复 LLM 覆盖破坏原经验；
- How Memory Management Impacts LLM Agents：相似经验会诱导相似输出，错误传播和“看似正确但不适合作经验”必须被控制。

GrowRAG 增量：同题 BASE/REUSE 配对、显式 benefit/harm 双账本和独立文档验证。

### 第 2 级：当前适用性硬约束

问题：即使历史可靠，它是否允许用于眼前的 `q0 + gap + environment`？

检查：

- 当前 gap category/slot/relationship 是否符合卡的 pattern；
- preconditions 是否满足、contraindications 是否触发；
- 当前 retriever 是否有卡要求的能力；
- corpus/document scope 与版本是否兼容；
- 模板是否只填当前实体，不会泄漏历史答案；
- 当前 repair stage 是否正确。

参考：

- RRM：task type、applicability conditions、required evidence、undesirable behaviors、query-adjustment pattern；
- ReFormeR：根据新 query 与首轮 top-k 检索结果选择历史 reformulation pattern；
- Useful Memories：保留具体来源，避免抽象记忆在不合适处强制覆盖。

GrowRAG 增量：把 applicability 与历史 reliability 明确分离，并将 gap、环境能力和禁用条件做成硬门。

### 第 3 级：结构软匹配与候选排序

问题：通过硬门的卡中，哪张最可能以较低成本填当前 gap？

特征：

- q0 与 query pattern 的语义/词项匹配；
- gap type、target role、slot 和 relation 方向；
- repair intent 与 transform form；
- DocumentSession 的精确文档局部提示；
- 原 query 和候选 query 的 pre-QPP；
- 预计额外检索/生成成本；
- 必须把 `KEEP/FRESH` 保留为候选，而不是只在卡片中选一个。

参考：

- QPP Query Variant Selection：检索前/后的 variant 质量预测；
- ReFormeR：pattern selection；
- RRM：query、异常与经验字段的匹配；
- ReMe：when-to-use/关键词式适配信息。

GrowRAG 增量：以 `gap × repair intent` 双轴匹配，并把 QPP 仅当特征，不把相似度或 QPP 误当可靠性。

### 第 4 级：执行后 evidence contract 核验

问题：这张卡不仅“看起来适用”，执行后是否真的完成了承诺？

检查：

- 是否获得目标 `gap_ids` 的新独立支持证据；
- need 是否向 `COVERED` 前进；
- 是否引入新冲突或高重复；
- 最终 answer claims 是否有支持；
- 相对 BASE/FRESH 的 paired outcome 是 beneficial、harmful 还是 neutral。

参考：

- S2G-RAG：gap closure 与 sentence-level evidence context；
- Self-RAG：文档相关性和生成内容支持判断；
- RRM：required evidence 字段。

GrowRAG 增量：ExperienceCard 自带可执行的 evidence contract；失约会回滚当前分支并更新卡片的动态可靠度。

## 9. 什么是“冻结 prompt 的 Judge”

### 9.1 Judge 是什么

Judge 是裁判/评估器。它不负责回答问题，也不负责想下一条 query，只做一项窄任务：

```text
给定原问题、原子 needs 和当前证据，
逐项判断 covered / partial / missing / conflicted，
引用 evidence IDs，输出 overall state 与 gaps。
```

同一个 LLM 可以在不同调用中扮演 Judge 和 rewriter，但实验上必须用不同 prompt、不同输出 schema、不同日志字段，避免“自己想搜什么，所以宣称证据缺什么”的循环偏差。

### 9.2 “冻结”具体冻结什么

- system prompt 和 user template 的精确文本；
- few-shot 示例及顺序；
- Judge 模型 ID/版本；
- decoding 参数；
- JSON schema；
- 后处理规则与阈值；
- `judge_prompt_version`。

在 dev 上确定后，test 期间一项都不改。若修改，则创建新实验版本并重新跑完整对照。

### 9.3 为什么冻结

1. 控制变量：改变的是经验复用策略，不是每组实验都换一套裁判；
2. 防止测试泄漏：不能看到 test 错例后继续“改 Judge prompt”；
3. 保证 episode 可比：卡片 reliability 的历史事件必须由同一判定协议产生；
4. 便于复现和消融：他人知道这个标签是怎样生成的。

### 9.4 v1 Judge prompt 草案

```text
System:
你是证据覆盖裁判。只能依据给定 Evidence Context 判断，
不得使用模型自身常识补全答案，不得生成下一条检索 query。
输出必须符合给定 JSON schema。

Input:
- original_query
- atomic_needs[{need_id, description, required}]
- evidence_units[{evidence_id, text, provenance}]
- previous_need_states / previous_gaps

Task:
1. 对每个 need 输出 COVERED/PARTIAL/MISSING/CONFLICTED；
2. COVERED/PARTIAL 必须列出 supporting evidence_ids；
3. CONFLICTED 必须列出冲突双方 evidence_ids；
4. 只为未解决 need 产生 gap；
5. 仅当所有 required needs COVERED 且无关键冲突时 overall=SUPPORTED。
```

正式 prompt 还需要加入“不要因答案看起来合理就判支持”“同一来源重复表述不能算独立证据”等负例，并在 dev 上测一致性与校准。

[Sufficient Context（ICLR 2025）](https://openreview.net/forum?id=Jjr2Odj8DJ) 是 Judge 定义的重要补充：模型可能在上下文不足时仍凭参数知识答对，因此“最终答案正确”不能反推“当前检索证据充分”；不完整、无法推出答案或相互矛盾的上下文应判不足。

## 10. Self-RAG 的 supported 与我们的判断是否相似

相似，但对象和时机不同。

[Self-RAG（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html) 训练模型生成 reflection tokens，主要包括：

- 是否需要 Retrieve；
- 文档对输入是否 Relevant；
- 生成片段是否 Fully/Partially/Not Supported；
- 输出整体是否 Useful。

它的 `IsSUP` 问的是“生成的内容 y 是否被文档 d 支持”。query 本身不是事实陈述，不能说 query 被文档 supported。

GrowRAG 推荐三道不同检查：

```text
执行前：ExperienceCard 是否可靠且适用于当前 gap？
执行后、回答前：新 query 是否履行 evidence contract，累计证据是否覆盖 q0？
回答后：answer claims 是否被当前证据支持？
```

Self-RAG 最直接启发第三道，也可为第二道的 relevance 检查提供标签设计；它没有解决历史 query repair 的跨题可靠迁移。

## 11. Query Transformation 的进一步定位

Query Transformation 是上位概念；建议从三个轴描述，而不是把所有词放在一个动作枚举中。

### 11.1 生成机制

- 规则/词典；
- LLM prompt rewrite；
- 训练型 rewriter；
- 从 ExperienceCard 实例化模板；
- retriever feedback/reranker feedback 驱动。

### 11.2 变换形式

- `PARAPHRASE`：同一信息需求换表达；
- `EXPAND`：加入相关术语、实体别名、上下文或伪文档；
- `DECOMPOSE`：拆成多个子查询；
- `ABSTRACT/STEP_BACK`：先问上位概念/原则；
- `HYDE`：生成假想文档作为 dense retrieval 表示；
- `CONSTRAIN/FILTER`：加入时间、实体、类型或来源限制。

### 11.3 修复目的

- 消歧；
- 补实体/属性/关系；
- 找 bridge entity；
- 提高词项召回；
- 扩大语义召回；
- 降低噪声；
- 验证冲突；
- 覆盖尚未回答的 sub-question。

一条 transform 可以同时有 `form=EXPAND`、`intent=DISAMBIGUATE`，并由 `generator=LLM_PROMPT` 产生。这个三轴表示有助于做经验匹配和消融，但分类法本身不是论文中心创新。

本轮新增精读重点：

- AIR：未覆盖项驱动的迭代改写与透明停止；
- Step-Back Prompting：抽象式 query transformation；
- MuGI：多伪参考 expansion 及 sparse/dense 融合；
- MMLF：sub-query + pseudo-document + late fusion；
- Q-DREAM：分解、子问题依赖和动态 passage 对齐；
- EACL 2026 Query Decomposition：以 exploration/exploitation 分配 sub-query 检索；
- PROGRAM：逻辑/时间/因果等 program type 与多 query、证据累积；
- EfficientRAG：无需每轮 LLM 调用的迭代 query 与过滤。

## 12. 中心贡献与前人“经验改写”的差异

### 12.1 最简洁的中心命题

> 对一个新的 RAG 问题，不仅检索相似的历史 query-repair 经验，还要分开验证“这张经验过去是否可靠”和“它现在是否适用”，执行后核验它承诺的证据缺口是否真正被填补，并显式最小化 BASE 原本正确却因复用变错的风险。

可使用的简单英文问题式标题：

> **When Should RAG Reuse a Query Repair?**

副标题或方法描述再写 safe/harm-aware transfer，不必在标题中使用 amortize。

### 12.2 与最接近工作的边界

| 工作 | 已经做了什么 | 没有覆盖的中心点 |
|---|---|---|
| ReFormeR | 从历史 reformulation 提取显式 pattern，并为新 query 选择/应用 | 没有跨独立题动态 benefit/harm 账本、BASE/REUSE 同题归因、执行后 evidence contract |
| RRM | query-only 程序经验、适用条件、required evidence、复用反馈、衰减/合并/裁剪 | 没有把 reliability 与 applicability 做成可校准的双门，也未以 `BASE correct→REUSE wrong` 配对风险为中心评价 |
| QPP variant selection | 预测/选择可能检索更好的 query variant | 不保存长期经验，不证明经验可靠或最终回答更好 |
| S2G-RAG / ReflectiveRAG | 同一道题内根据当前证据继续 FRESH 修复并停止 | 不跨题复用历史 query repair |
| Self-RAG / SIM-RAG | 相关性、支持性或 Accept/Reject critic | 不管理历史 repair transfer 的可靠度与当前适用性 |
| Useful Memories / ReMe | 原始与抽象经验、经验生命周期和错误传播问题 | 不是冻结 RAG 上 q→q′→evidence 的同题安全迁移协议 |

### 12.3 如果只做系统拼接会怎样

如果论文只写“三层记忆 + QPP + gap + controller + query actions”，确实会像裁缝式组合。要把贡献变成可检验的新研究问题：

1. 定义并发布 `BASE / REUSE / FRESH` 同题、同预算的 paired evaluation protocol；
2. 用 `reliability × applicability × evidence contract` 选择性调用协议减少 harmful transfer；
3. 把 `P(REUSE wrong | BASE correct, REUSE called)`、helpful transfer、repair regret 与调用成本列为主指标；
4. 证明每一道门都不可替代：去掉 paired ledger、合并 reliability/applicability、去掉 contract、去掉 rollback 分别会怎样；
5. 证明 BASE 变强时调用率自然下降，但每次调用的净收益上升，而非追求高复用率。

这是目前较可守的中心贡献，但投稿前仍需持续跟踪 RRM、ReFormeR 及 2026 年后续版本，不能现在声称“首次”。

## 13. 下一轮需要确认而不是现在写代码的事项

1. `CONFLICTED_OR_UNCERTAIN` 是否正式采用三动作：`FRESH_VERIFY / ROLLBACK / STOP_ABSTAIN`；本文建议采用。
2. 是否接受主实验中 test memory 全冻结、在线自进化另列协议；本文建议采用。
3. reliability 是否采用 benefit 下界 + harm 上界，而 EMA 只做近期趋势；本文建议采用。
4. Judge 是否严格只判断 need coverage，不生成 query；本文建议采用。
5. 回答后的 Self-RAG 风格 claim support check：MVP 可开关，正式论文建议开启并单独消融。
6. 最大 repair 默认 3、dev 在 1/2/3/4 中校准；本文建议采用。

确认这些设计后，再统一升级 EvidenceState、EvidenceLedger、ExperienceCard reliability ledger 与 controller schema，避免讨论期间反复重写代码。
