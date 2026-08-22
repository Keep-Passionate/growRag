# 单 Query Episode 与长期经验卡 Schema v0

更新日期：2026-08-22
状态：Schema v0 已落地；实现允许增加审计字段，但不能删掉核心语义

## 1. 三个对象，不混为一谈

### QueryEpisode

一次问题从开始到结束的不可变检索轨迹。它可以检索一篇或多篇文档，所以 episode 不是“单文档”。

### DocumentSession

可选的归组容器。例如用户正在围绕一篇论文连续提问，它只记录这一组 QueryEpisode 的 ID。文档工作结束后冻结容器，但每个问题回答完成时就先冻结自己的 episode。

### ExperienceCard

从已经验证有增益的 QueryEpisode 中抽取、可跨问题复用的长期程序经验。它说明“什么状态下用什么查询修复”，不保存历史答案作为当前题事实。

## 2. QueryEpisode：最小但充分

推荐规范：

```yaml
QueryEpisode:
  schema_version: query_episode.v0
  episode_id:
  document_session_id: optional
  run_context_id:
  comparison_group_id:
  original_query:
  opened_at:
  frozen_at:

  turns:
    - turn_id:
      action: BASE | REUSE_REPAIR | FRESH_REPAIR

      query_action:
        operator:
        executed_queries: []  # v0 强制每个 turn 恰好执行一条
        target_gap_ids: []
        experience_card_version: optional

      evidence_refs:
        - doc_id:
          unit_id:
          rank:
          retrieval_score:
          content_hash:

      retrieval_state:
        sufficiency: sufficient | insufficient | unknown
        gaps:
          - gap_id:
            category:
            target:
            slot:
            description:
        progress:
          new_evidence_count:
          gaps_closed: []
          duplicate_ratio:

      decision:
        next_action: ANSWER | REUSE_REPAIR | FRESH_REPAIR | STOP_ABSTAIN
        reason_code:

      cost:
        retrieval_calls:
        input_tokens:
        output_tokens:
        latency_ms:

  result:
    final_answer_ref:
    final_answer_hash:
    final_evidence_refs: []
    terminal_reason: sufficient | baseline_complete | no_progress | budget | risk_reject | error
```

### 每一项为什么保存

| 字段 | 理由 |
|---|---|
| `original_query` | 它是实验对象；没有原问题就无法判断改写是否漂移 |
| `run_context_id` | 用一个指针关联语料、切块、索引、retriever、reranker、LLM、judge 和 prompt 版本，避免每条 episode 重复几十个变量 |
| `comparison_group_id` | 把同一原问题的 BASE/REUSE/FRESH 反事实分支绑定在一起，防止跨题误比较 |
| `action` | 区分原始 RAG、历史经验修复和现场修复 |
| `executed_queries` | 记录真正送给检索器的 `q'`；v0 每个 turn 只允许一条，多个 variant 必须分成多个分支/turn，才能把证据、成本和收益归因到具体改写 |
| `operator` | 说明变化的语义目的，例如消歧、分解、HyDE，而不只是字符串差异 |
| `target_gap_ids` | 说明为什么执行这次改写，并允许判断它是否真的填补目标缺口 |
| `experience_card_version` | 精确追溯哪一版长期卡影响了本轮动作 |
| `evidence_refs` | 能复查检索结果，但不在记忆库复制整篇文档 |
| `sufficiency + gaps` | 显式记录“还缺什么”，使继续搜索可解释 |
| `progress` | 支撑边际增益/无进展停止；保留原始小量后可重新校准阈值 |
| `cost` | adaptive controller 必须判断收益是否值得额外调用 |
| `terminal_reason` | 区分真的证据充分与“搜不动、没预算、风险拒绝” |

### 不保存完整 CoT

不保存隐藏的逐字 Chain-of-Thought。它长、难验证、可能泄漏历史事实，而且更换模型后不可复现。需要保存的是可观察、可审计的结构化状态：gap、动作、执行查询、证据指针、判断理由和结果。

以后若研究“压缩 CoT 笔记”，也只使用以下可验证片段：

- 失败类型；
- 缺失的实体/属性/关系；
- 执行的检索动作；
- 哪项新证据关闭了哪个 gap；
- 哪个停止条件触发。

## 3. RunContext：解决“环境变量太多”

环境不要求塞入每条经验。单独建立不可变注册表：

```yaml
RunContext:
  run_context_id:
  created_at:
  corpus_id:
  corpus_version:
  chunk_index_policy_id:
  retriever_family:
  retriever_id:
  reranker_id: optional
  generator_id:
  state_prompt_version:
  repair_prompt_version:
  judge_id:
  budget_policy_id:
```

Episode 只保存 `run_context_id`。

v1 不学习跨所有环境泛化：固定一个 BASE 环境，只有完全相同的 context 才允许经验进入主实验。之后再分两级：

- 硬兼容：语料版本、索引/切块策略和 retriever family；
- 软兼容：模型大小、prompt 小版本、top-k 等，在开发集上单独验证。

长期卡只保存自己需要的能力条件，例如 `dense_embedding`、`graph_neighbor_expand`，而不是枚举几十个配置变量。

## 4. 完成即冻结，验证结果采用追加事件

每个 QueryEpisode 在回答或弃答后立即冻结，无论结果正确还是错误。冻结表示“当时发生过的事实不再被覆盖”，不是“这条经验可信”。

Gold、人评或 judge 可能晚到，因此验证写成只追加的事件：

```yaml
VerificationEvent:
  verification_id:
  episode_id:
  target: answer | evidence | paired_benefit
  source: gold | human | judge | proxy
  metric_id:
  score:
  verdict: pass | fail | uncertain
  direct_baseline_episode_ref: optional
  baseline_score: paired_benefit 必填
  treatment_score: paired_benefit 必填
  quality_threshold: paired_benefit 必填
  created_at:
```

这样 episode 不被重新改写，但可以追加新证据。配对事件只有在以下条件同时成立时才可归档：同一原问题、同一 `comparison_group_id`、同一冻结 `RunContext`/预算；baseline 必须是只执行一次原查询的 BASE episode；treatment 必须实际包含 REUSE 或 FRESH。事件会据两个分数和质量阈值自动核对 `benefit / neutral / correct-to-wrong harm`，不能手写一个“有收益”标签蒙混过去。

### 为什么失败 episode 也要冻结

- 如果删掉失败，就无法统计 harmful reuse；
- 只留成功会产生幸存者偏差；
- 后续训练 controller 需要知道哪些状态和动作会失败；
- 延迟 gold 可能把一条暂时 unknown 的 episode 改判。

这不违反“v1 只使用成功经验”：失败 episode 进入隔离冷档案，绝不参与在线召回。

## 5. 成功分三级

“修复后答对”并不等于“这条经验产生了收益”。DIRECT 也可能答对。

1. `working_success`：修复后达标，但缺少 DIRECT 对照；
2. `beneficial_success`：相同环境和预算下优于 DIRECT，或补回 DIRECT 缺失的 gold supporting facts；
3. `trusted_success`：在独立问题/独立文档上多次产生收益，且 `DIRECT 正确 -> REUSE 错误` 的风险低。

只有第 2 层可以生成 candidate card；只有第 3 层可以 active。

验证来源强度：

- Gold answer、gold supporting facts 或人工确认：可参与晋升；
- 多个独立 judge 且意见一致：先留 candidate，不能直接 trusted；
- 只有相似度、Recall、QPP 或 sufficiency proxy：只留局部，不作为 trusted 证明。

## 6. DocumentSession：只归组，不再总结一遍

```yaml
DocumentSession:
  schema_version: document_session.v0
  session_id:
  scope_ref:
  scope_version:
  episode_ids: []
  opened_at:
  frozen_at:
```

它的用途：

- 计算一条修复是否只在同一文档的近重复问题中成功；
- 晋升时统计独立文档数；
- 支持用户所说的“单文档工作结束后冻结”。

v1 不在 DocumentSession 内维护一份由 LLM 反复覆盖的文档总结，否则会重新制造冗余、版本冲突和 Useful Memories 指出的 consolidation 损坏。

## 7. ExperienceCard：最小长期服务结构

```yaml
ExperienceCard:
  schema_version: experience_card.v0
  card_id:
  version:
  created_at:
  parent_versioned_ids: []
  lifecycle_state: candidate | active | quarantine | retired

  activation:
    stage: pre_retrieval | post_retrieval
    query_pattern:
    gap_pattern:
    preconditions: []
    contraindications: []
    required_retriever_capabilities: []

  repair:
    operator_type:
    slot_template:
    evidence_contract:

  provenance:
    source_episode_turn_refs: []
    canonical_example_refs: []
    independent_episode_count:
    independent_document_count:

  validation:
    verification_tier:
    verification_event_ids: []
    matched_trials:
    benefit_count:
    neutral_count:
    harm_count:
    direct_correct_trials:
    mean_gain:
    last_validated_at:

  serving:
    expected_cost:
    use_count:
    last_used_at:

  activation_policy:
    allowed_verification_tiers: [gold, human]
    min_independent_episodes:
    min_independent_documents:
    min_matched_trials:
    min_benefit_count:
    max_conditional_harm_rate:
```

### 每组元素为什么需要

#### identity / version / state

经验不能被 LLM 原地覆盖。抽象、合并或修正都生成新版本，通过 `parent_versioned_ids` 保留版本谱系；旧版可回滚。`quarantine` 表示暂时停用，证据仍保留。

#### activation

- `query_pattern`：哪类问题；
- `gap_pattern`：哪种当前缺口；
- `preconditions`：RRM 风格适用条件；
- `contraindications`：明确禁止条件，防止安全阈值过松时误用；
- `required_retriever_capabilities`：这项修复要求底座具备什么能力。

当前问题算出的 `applicability=0.83` 不写回卡片。它是当前 query × 当前 gap × 当前环境的瞬时结果，每次调用都应重算。

#### repair

- `operator_type`：`rewrite / expand / hyde / decompose / disambiguate / add_constraint / bridge / focus`；
- `slot_template`：怎样把当前题的实体、关系和约束填入算子；
- `evidence_contract`：执行后必须新增什么证据，才算这条经验履约。

例如：“若缺少人物出生国家，修复后至少应新增同时连接该人物与出生地/国家属性的证据。”这比“再搜一次”更可验证。

#### provenance

长期卡必须能指回具体 `q -> q' -> evidence -> outcome`。`independent_document_count` 防止同一文档的十个近重复问题被误当成十次独立成功。

#### validation

不要只保存一个 mutable confidence。保留 benefit/harm 的充分统计量，并用 `verification_event_ids` 指回每次真实配对事件，之后可以重新计算 Wilson 上界、Bayesian posterior、EMA 或其他估计。

核心风险事件：

```text
BASE/DIRECT 达标，但 REUSE_REPAIR 不达标
```

#### serving

只保留最小成本画像。adaptive routing 不仅问“可能有用吗”，也问“值得多花这次检索、token 和延迟吗”。

#### activation_policy 与可信注册边界

安全阈值不藏在代码里，也不把“一次成功”直接当作 active。每张卡记录当时采用的晋升政策：允许的验证来源、最少独立题/文档、最少配对试验和最大 conditional harm。第一版默认 gold/human、至少两个独立 episode/文档、至少两次收益，并限制 observed harm；这些阈值之后可在开发集校准和做宽松/严格消融。

`ExperienceCard` 只是不可变记录；只有通过 `ExperienceCardRegistry` 校验后才算 serving card。Registry 会核对：来源 episode/turn 真实存在且是 repair、来源文档计数真实、每个来源有 paired benefit、验证事件计数和 mean gain 可由归档重算、父版本存在且同一版本不能覆盖。

## 8. 为什么不只保存历史改写字符串

答案不是“不保存”，而是保存位置不同：

```text
冻结 QueryEpisode：具体 q -> q'、证据、结果
                  ↓ source pointer
版本化 ExperienceCard：抽象 operator、条件、禁用项、证据契约
```

- 只存具体改写：容易把历史实体和措辞机械搬到新题，迁移范围太窄；
- 只存抽象卡：容易丢掉例外和来源，反复总结后无法验证；
- 双表示：服务时使用小卡，出错时可回到原 episode 审计或重建。

长期卡可以引用 1–3 个 canonical concrete examples 作为应用示例，但不复制历史答案或文档正文。

## 9. 明确不保存什么

- 不复制 full document/full chunk；只存 ID、rank、score、hash；
- 不保存完整隐藏 CoT；
- 不把历史答案、选项或事实结论写进 active card；
- 不把 embedding 当规范数据；它只是带 encoder version 的可重建缓存；
- 不把所有未执行的 query variant 塞进主 episode；候选全集放实验日志指针；
- 不把当前题的 applicability 写成卡片永久属性；
- 不用一个不可解释的 confidence 覆盖原始计数。

## 10. 晋升流程

```text
OPEN QueryEpisode
  -> 回答结束，FROZEN
  -> 追加 gold/human/judge 验证
  -> verified beneficial success
  -> candidate card（仍指向来源 episode）
  -> 独立 query/document 上配对验证 benefit、harm、cost
  -> ACTIVE 或 QUARANTINE
```

规则：

- 单次成功只能成为 exemplar/candidate；
- 合并总是生成新 card version；
- source episode 永不因 consolidation 删除；
- v1 只有 verified beneficial success 可上线；
- 失败 episode 只做审计与训练负例。

## 11. 与前人记忆结构的差异

### GAM-RAG

GAM-RAG 的最小记忆单位是语料中的句子。每句保存 task/time memory vector 及两个不确定性项；LLM judge 给 sentence support 后，用 Kalman-inspired gain 沿 query embedding 更新句向量和不确定性，最终改变图传播与排序。

它不保存 q→q'、修复算子、适用/禁用条件、raw query episode lineage 或 DIRECT→REUSE 配对伤害。其 gain update 可作为未来卡片优先级更新的对照，但不是我们的 schema 基线。

### RRM

RRM 是最近邻：已经有 task type、applicability conditions、required evidence、retrieval strategy、undesirable behavior、query-adjustment pattern，且采用 query-only reuse、成功/失败双库和生命周期衰减。

不能把这些字段本身称为创新。GrowRAG 要检验的增量是：

- 永久保留具体单-query episode，而不是只留压缩程序记录；
- 具体改写与抽象卡双表示；
- 版本化、可回滚的晋升；
- 相同预算下 BASE/REUSE/FRESH 配对结果；
- 历史可靠性与当前适用性分别校准；
- 显式控制正确到错误的复用伤害。

### SegMem-RAG

SegMem-RAG 已有 episodic/procedural/semantic memory，并根据经验在多个语料源之间路由。它证明“episode + 程序记忆 + 自适应路由”本身不是空白。

它主要回答“当前题去哪一个 corpus/source 检索”；GrowRAG 回答“当前证据缺口应否调用一条历史 query repair，以及怎样避免负迁移”。

### Useful Memories

它证明 raw episodic traces 与 consolidated abstractions 不应相互替代，且持续覆盖更新会把好记忆改坏。GrowRAG 的具体—抽象双表示、只追加验证和版本化晋升是对这条警告的工程化响应，但“双记忆”本身不是创新。

### S2G-RAG / Skill-RAG

S2G 已有 sufficiency 与 `category/target/slot/description` gap；Skill-RAG 已将失败状态分型并路由到 rewrite/decompose/focus/exit。因此结构化 gap 和 typed repair 不是创新。我们的研究对象是这类本题修复怎样成为可追溯、可验证、可拒绝的跨题经验。

## 12. 必须做的消融

为了回答“是否只是已有组件拼接”，至少比较：

1. raw episode only；
2. abstract card only；
3. dual representation；
4. 覆盖式合并 vs 版本化晋升；
5. reliability only；
6. applicability only；
7. reliability + applicability；
8. 去掉 harm gate；
9. BASE / nearest-history / always-reuse / FRESH；
10. 同文档、跨文档和 strict cross-entity transfer。

主报告同时给出 benefit、conditional harm、coverage、answer quality、retrieval quality 和 cost。没有实验前只能说这些是设计优势，不能声称已经优于 GAM-RAG、RRM 或 S2G-RAG。
