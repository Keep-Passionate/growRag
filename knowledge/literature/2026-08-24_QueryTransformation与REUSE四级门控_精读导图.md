# Query Transformation 与 REUSE 四级门控：精读导图

更新日期：2026-08-24  
用途：解释 GrowRAG 如何使用 query transformation、四级 REUSE 选择分别来自哪些工作，以及下一批论文的精读顺序。

## 1. Query Transformation 在 GrowRAG 里的位置

它不是一个单独的“万能改写模型”，而是 controller 在证据不足时执行的动作族：

```text
Evidence gap
  -> 决定 KEEP 还是 TRANSFORM
  -> 选择变换形式 form
  -> 选择修复目的 intent
  -> 由 FRESH prompt 或 ExperienceCard 产生具体 query
  -> 检索
  -> 用 evidence contract 检查是否真的补缺口
```

推荐用三个正交维度描述：

| 维度 | 例子 | 回答的问题 |
|---|---|---|
| generator | rule、LLM prompt、trained rewriter、ExperienceCard template | 谁/什么机制生成改写？ |
| transform form | paraphrase、expand、decompose、step-back、HyDE、constrain | query 在形式上发生了什么？ |
| repair intent | disambiguate、bridge entity、fill attribute/relation、verify conflict、evidence focus | 为什么要这样改？ |

因此用户之前的理解基本正确：expansion、decomposition、paraphrase 都属于广义 rewrite/transformation 的具体形式；“LLM rewrite”说的是生成机制，不是与 expansion 平级的唯一动作。

## 2. 建议保留的首版动作

为了避免一开始动作过多，v1 推荐：

```yaml
QueryDecision: KEEP | TRANSFORM

transform_form:
  PARAPHRASE
  EXPAND
  DECOMPOSE
  CONSTRAIN

repair_intent:
  DISAMBIGUATE
  BRIDGE_ENTITY
  FILL_ATTRIBUTE
  FILL_RELATION
  VERIFY_CONFLICT
  EVIDENCE_FOCUS
```

HyDE 先作为 dense-retrieval 专用基线，不与字符串 query 混在同一执行接口；Step-Back 先作为 `ABSTRACT` 扩展动作；pseudo-document 与 multi-query late fusion 在首版管线跑通后加入。

### 2.1 Gap 到动作的默认映射

| 当前 gap | 默认 intent | 优先 form | 原因 |
|---|---|---|---|
| 实体歧义 | DISAMBIGUATE | CONSTRAIN / EXPAND | 加类型、别名、上下文，减少同名噪声 |
| 缺桥接实体 | BRIDGE_ENTITY | DECOMPOSE | 先查桥，再用桥查目标 |
| 缺属性 | FILL_ATTRIBUTE | CONSTRAIN / PARAPHRASE | 明确实体与属性槽位 |
| 缺关系 | FILL_RELATION | DECOMPOSE / PARAPHRASE | 明确关系方向和两端实体 |
| 关键词不匹配 | EVIDENCE_FOCUS | EXPAND | 增加别名/术语帮助 BM25 或 dense retrieval |
| 证据冲突 | VERIFY_CONFLICT | CONSTRAIN | 指定时间、版本、来源类型，查独立证据 |

这只是默认规则。实验必须保留 `KEEP` 和 `FRESH`，否则 controller 会被迫使用历史经验。

## 3. 一条安全 Transform 应输出什么

```yaml
QueryTransform:
  original_query_id: q0
  source: REUSE           # BASE | FRESH | REUSE
  experience_card_ref: card_17@4
  target_gap_ids: [gap_2]

  generator: CARD_TEMPLATE_WITH_LLM
  form: EXPAND
  intent: DISAMBIGUATE

  preserved_need_ids: [need_1]
  anchor_evidence_ids: [ev_12]
  added_constraints:
    - type: entity_type
      value: "film director"
      evidence_ref: ev_12

  generated_queries:
    - "..."

  evidence_contract:
    target_need_ids: [need_2]
    expected_relation: "..."
    forbidden_history_facts: true
```

关键约束：

- 原问题 `q0` 永久不改，transform 只是临时检索动作；
- 新增的具体实体/约束必须来自当前证据 anchor，或是空槽位模板，不能把历史答案带入；
- 每条 query 明确针对哪些 gap；
- REUSE 只提供程序规则，不直接提供旧题事实；
- 执行后能检查履约，而不是只凭 query 文本看起来更好。

## 4. 如何评价一个 Transform

不要只问“改写句子是否流畅”。至少分五层：

1. **Intent faithfulness**：是否仍服务原问题，是否错误加入旧题实体/答案；
2. **Retrieval quality**：gold document/fact Recall@k、MRR/nDCG 或 qrels；
3. **Evidence progress**：新独立证据、gap closure、coverage 增量、冲突变化；
4. **Answer effect**：相对 BASE 的 EM/F1/claim support 变化；
5. **Safety/cost**：`BASE correct→REUSE wrong`、额外调用、token、延迟。

QPP 主要预测第 2 层，不能替代第 3–5 层。最终卡片可靠度必须由 paired outcome 更新。

## 5. Query Transformation 精读链

### 直接重合风险：先读这组，再谈“改写创新”

下列工作已经覆盖了“多种改写动作、自适应选择、失败反馈、成功轨迹学习”等常见想法。它们说明动作词表本身不能作为 GrowRAG 的中心贡献：

1. **DMQR-RAG** — arXiv 预印本：四种改写策略、adaptive selection、`KEEP` 与多查询融合。它直接否定“我们首次把改写拆成动作并选择”的表述。
2. **SAGE** — arXiv 预印本：strategy experts 与强化学习选择。它覆盖“用学习控制器选择改写策略”的宽泛命题。
3. **Think Then Rewrite** — AAAI 2026 Main：先推理和消歧，再生成检索 query。它覆盖“先显式分析失败原因再改写”的一部分，但不做跨问题经验可靠性迁移。
4. **ReFeed** — AAAI 2026 New Frontiers in IR Workshop：从失败检索反馈生成下一条 query，并只用成功例训练。它与 FRESH repair 很近，但没有 GrowRAG 的历史卡片双门控、同题伤害账本和回滚契约。
5. **MaFeRw** — AAAI 2025 Main：用多方面反馈奖励训练 rewriter。它告诉我们不能只凭最终答案给改写打分。
6. **AdaQR** — EMNLP 2024 Main：把边际答案概率作为弱反馈并用偏好优化训练 query rewriter。
7. **RetPO** — Findings NAACL 2025：从 retriever preference 构造大规模改写偏好数据；适合作为以后训练小 rewriter 的对照。
8. **SELF-multi-RAG** — Findings EMNLP 2024：联合决定何时检索、如何改写以及证据是否相关；是 controller 的强近邻，但不维护跨题的动态经验可靠性。

因此当前差异化必须是：**不是再发明一种 query rewrite，而是决定一条历史 repair 何时可以安全迁移，并在执行后用证据契约验证、回滚和动态更新其可靠性。**

### 基础层：先理解“改写、扩写、表示变换”

1. **Query Rewriting in Retrieval-Augmented Large Language Models** — EMNLP 2023 Main，已在旧包  
   学习：训练/提示 rewriter 如何把问题改成更适合检索的 query；它是所有 rewrite baseline 的基础。
2. **Query2doc** — EMNLP 2023 Main，已在 08-22 包  
   学习：用 LLM 生成 pseudo-document 做 expansion，为什么扩写可改善 lexical/dense retrieval。
3. **HyDE** — ACL 2023 Long，已在 08-22 包  
   学习：生成假想文档并编码，不应简单理解成“换一句 query”。
4. **Exploring the Best Practices of Query Expansion with LLMs / MuGI** — Findings EMNLP 2024，新 RDF  
   学习：多 pseudo-reference、原 query 与扩写的平衡、sparse/dense 融合；提醒“更多文本”不一定更好。
5. **Take a Step Back** — ICLR 2024，新 RDF  
   学习：通过抽象/上位原则构造新的检索或推理视角；对应未来 `ABSTRACT` action。

### 多查询与分解层

6. **AIR** — ACL 2020 Main，新 RDF  
   学习：只围绕尚未被 justification 覆盖的 query terms 继续检索，以及无新 terms 时停止。
7. **Q-DREAM** — ACL 2025 Long，新 RDF  
   学习：分解不是独立子问题列表，还存在 dependency 与动态 passage 对齐。
8. **Query Decomposition for RAG: Balancing Exploration-Exploitation** — EACL 2026 Long，新 RDF  
   学习：多个 sub-query 之间如何分配继续探索/利用已有方向的检索预算；不是本项目 v1 的 controller，但可用于 decomposition baseline。
9. **MMLF** — Findings NAACL 2025，新 RDF  
   学习：sub-query、pseudo-document、独立检索和 reciprocal rank fusion 的组合；说明 transform 之后还要设计结果融合。
10. **PROGRAM** — Findings ACL 2026，新 RDF  
    学习：把 logical/temporal/causal 等 program type 用于多 query 与证据累积；警惕把 action taxonomy 当作新贡献。
11. **EfficientRAG** — EMNLP 2024 Main，新 RDF  
    学习：迭代 query 和过滤不必每轮调用大 LLM；为未来小控制器/轻量 rewriter 提供基线。

## 6. REUSE 四级门控的论文来源图

```text
候选卡
  │
  ├─ 1 Reliability hard gate
  │     RRM / ReMe / GAM-RAG / Useful Memories /
  │     How Memory Management Impacts LLM Agents
  │
  ├─ 2 Applicability hard constraints
  │     RRM applicability conditions + ReFormeR pattern scope
  │
  ├─ 3 Structured soft match & rank
  │     ReFormeR selection + QPP variant selection + RRM/ReMe metadata
  │
  └─ 4 Post-execution evidence contract
        S2G-RAG gap closure + Self-RAG relevance/support + RRM required evidence
```

### 6.1 第 1 级：历史可靠性

必须回答“过去有没有相对 BASE 的独立净收益和伤害”，而不是“这张卡曾经成功”。

- RRM：反馈、频次、时间衰减、合并/裁剪；
- ReMe：正负程序经验与 utility 生命周期；
- GAM-RAG：gain 与不确定性/近期更新；
- Useful Memories：持续覆盖会损坏记忆；
- How Memory Management Impacts LLM Agents：相似输入会复制历史行为，错误会传播，未来任务结果可反过来成为经验质量标签。

GrowRAG 的增量：不可变 paired ledger、benefit 下界、harm 上界、独立文档计数和动态隔离。

### 6.2 第 2 级：当前适用性

必须回答“这条历史规则是否允许用于眼前的 gap 和环境”。

- RRM：task type、applicability conditions、required evidence、undesirable behavior、query-adjustment pattern；
- ReFormeR：利用新 query 与首轮检索结果选择 reformulation pattern。

GrowRAG 的增量：将它与 reliability 分离，并加入 gap category/slot、contraindications、retriever capabilities、document/corpus version 与 query-only 防泄漏硬门。

### 6.3 第 3 级：软匹配与排序

- ReFormeR：pattern selector；
- QPP Query Variant Selection：pre/post-retrieval effectiveness prediction；
- RRM/ReMe：query、异常、when-to-use 等描述。

GrowRAG 的增量：匹配 `query pattern × gap pattern × repair intent`，QPP 只做候选特征，并始终保留 KEEP/FRESH。

### 6.4 第 4 级：执行后核验

- S2G-RAG：证据累计、gap 到 query、充分性；
- Self-RAG：document relevance 与 answer support；
- RRM：required evidence。

GrowRAG 的增量：卡片自带 evidence contract；只有新证据确实关闭目标 gap 才算履约，失约会回滚当前分支并动态更新 reliability。

## 7. 需要特别防止的“看似适用”

1. **语义近但关系方向相反**：`A 的父亲` 与 `A 的儿子` embedding 很近；
2. **问法相同但属性不同**：出生地经验不能迁移到药物副作用；
3. **同文档有效、跨文档失效**：局部 alias/section 规律不能晋升为通用卡；
4. **旧环境有效**：依赖 graph neighbor expansion 的卡不能给只有 BM25 的环境；
5. **历史答对但无增益**：BASE 本来也答对，不能给卡记 beneficial；
6. **query 好看但证据没变**：QPP 或语言流畅不能替代 gap closure；
7. **强模型自行补全**：答案正确不代表证据充分，必须看 gold supporting facts/claim support。

## 8. 推荐阅读顺序（只列当前方向必需）

假设旧清单已读一半，下一轮不要从头重读。建议：

1. Sufficient Context（新，ICLR 2025）
2. HALT（新，2026-08 预印本，最新停止重合风险）
3. How Memory Management Impacts LLM Agents（新，ACL 2026 Long）
4. DMQR-RAG → SAGE → Think Then Rewrite → ReFeed（先排除“动作选择/反馈改写”重合）
5. 回看 RRM 的 applicability/required evidence/feedback/lifecycle
6. 回看 S2G-RAG 的 Evidence Context、Gap、停止和训练数据部分
7. AIR（新，ACL 2020 Main）→ SIM-RAG（新，SIGIR 2025）
8. MaFeRw → AdaQR → RetPO → SELF-multi-RAG（反馈、训练和 controller 对照）
9. 回看 ReFormeR 的 pattern extraction/selection
10. MuGI → Q-DREAM → EACL 2026 Query Decomposition
11. MMLF → PROGRAM → EfficientRAG
12. TASR → Stop-RAG → When Should Multi-Round RAG Stop（停止基线与方法学警告）

Self-RAG、Query Rewriting、Query2doc、HyDE 已读过则只按上述问题回查对应章节，不必整篇重读。

## 9. 当前研究边界

不能作为中心创新：

- query rewriting/expansion/decomposition/HyDE；
- 结构化 gap；
- 累计 sentence evidence；
- need/hop coverage stopping；
- QPP 选 variant；
- applicability conditions；
- 衰减、merge、forget、top-k。

当前可守的中心增量：

> 在冻结 BASE RAG 上，用可回溯的结构化证据状态约束历史 query repair 的迁移；分开动态历史 reliability 与当前 applicability，执行后核验 evidence contract，并以同题 paired outcome 显式限制 harmful reuse。

这一定义将 Query Transformation 变成被验证的“动作”，而不是凭历史相似度直接注入的“答案提示”。
