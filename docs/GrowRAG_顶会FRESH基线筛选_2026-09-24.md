# GrowRAG：按顶会要求重新筛选 FRESH 基线

核验日期：2026-09-24。AI 辅助定向检索与代码审计，不是穷尽性综述。推荐不等于用户已确认；本轮没有运行新实验。

## 先看结论

**最贴近我们修复机制的候选：S2G-RAG（ACL 2026 主会长文）。最方便先做无需微调的多轮对照：DualRAG（ACL 2025 主会长文，非 FT 版本）。**

两篇都包含本题内部的查询生成/变换，不依赖跨题经验库，都涉及 HotpotQA。但必须承认：S2G 需要训练过的判断器；DualRAG 是较完整的多轮系统，不是一个纯粹的检索前改写模块。不能宣称它们同时满足“纯前置、免训练、极轻量”的全部偏好。

如果限定为“2025—2026、CCF A 主会长文、专门研究一次检索前 query rewrite、无需训练、公开可直接复现、HotpotQA 原生实验”，本轮尚未核实到全部满足的论文。这是检索边界，不是证明不存在。

## 筛选标准

- 将用户的“A区”暂按 **CCF A 类**理解；若指学校自己的 A 类目录，需要另行核对。
- 会议看正式长文/Full/Regular；不把 Findings、Workshop、短文混算为主会 A 类长文。
- 中科院一区与 JCR Q1 是不同的期刊分区体系，不能套在 ACL/AAAI 会议上；期刊还要注明分区年份、学科口径。
- 本项目 FRESH 指**不使用跨题历史经验、本题现场生成检索动作**。它不自动等于“无需参数训练”，也不意味着只检索一次。
- 除发表等级外，还检查问题形式、检索语料、公开代码/权重、训练门槛和许可。

评级依据：[CCF 2026 第七版发布说明](https://www.ccf.org.cn/Academic_Evaluation/By_category/)；[人工智能目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)。官方说明明确排除 Short、Findings、Workshop 等形式。人工智能目录中 ACL、AAAI 列在 A 类。

## 1. S2G-RAG：方法最贴近，但有判断器训练门槛

[正式论文](https://aclanthology.org/2026.acl-long.1185/)：*S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA*。ACL 2026 主会长文，DOI `10.18653/v1/2026.acl-long.1185`。

核心是：本题累计证据 → 判断是否足够 → 明确缺什么 → 将缺口映射为下次查询。它使用句级证据上下文，实验包括 HotpotQA、TriviaQA、2WikiMultiHopQA。这里的证据记忆是本题内状态，不是我们研究的跨题修复经验库。

值得精读的问题：

1. “证据相关”和“足以回答”如何区分？
2. 缺口如何结构化，怎样变成查询，而不是笼统要求 LLM 重写？
3. 哪些证据被保留，什么时候停止？
4. 训练判断器的标签从何而来，哪些实验贡献来自判断器，哪些来自迭代检索？

[作者代码](https://github.com/nianaaa/S2G-RAG)，核验提交 `5d842a67a0a99a7b545bbad0dc402ceaae0e5eff`。公开推理入口需要 Llama-3.2-3B 判断器及训练后的 LoRA；`--gpt` 替换回答/证据提取，不免除判断器。检查仓库未找到成品 adapter 下载；这不等于作者绝对没有其他发布渠道。训练脚本公开，但尚未在本机执行。

因此，现有 GrowRAG 的 `S2G_GAP2` 只是启发式适配，**不能标为完整 S2G-RAG 复现**。若用固定 Qwen judge 替代，必须叫同底座适配版，并单列这一差异。原版复现需先解决权重或训练路径。当前未发现仓库 LICENSE，不直接把作者源代码复制并推送到我们的 GitHub。

## 2. DualRAG：无需微调版本可做工程对照，但范围更广

[正式论文](https://aclanthology.org/2025.acl-long.1539/)：*DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering*。ACL 2025 主会长文，DOI `10.18653/v1/2025.acl-long.1539`。

它一边推理、定位知识需求并生成查询，一边整理本题检索知识。**本题知识整理不等于跨题经验复用。** 它同时研究查询、重排序、知识整理，因此不是纯 query-rewrite baseline；适合我们的多轮 FRESH 系统级对照。

原文区分 **DualRAG** 和 **DualRAG-FT**，前者无需新增微调，后者才训练。原文使用 Qwen2.5-72B/7B，并包含 HotpotQA、2Wiki、MuSiQue。[原文方法和实验](https://arxiv.org/html/2504.18243v1)。使用不同 Qwen 快照得到的是受控迁移结果，不能直接声称复现论文表格。

精读：§3.1 的两个过程、§3.2 与非 FT 版本的区别、§4 数据/检索/评价、附录 A 的提示词。尤其要理解：查询生成改进和证据整理改进不能全部算作“改写”的收益。

[作者代码](https://github.com/cbxgss/rag)，核验提交 `349f9175b2c72deea8a1f510dcc769ce50ab0f85`。重点文件为 `src/rag/duralrag/prompt.py`、`rag.py`（作者目录确实拼作 duralrag）。当前代码默认模型与论文不同，有无限重试配置和提示示例二次赋值；真正复现前须记录实际生效值。未发现 LICENSE，不直接复制发布。

## 3. 够新、级别合格，但不作为本轮主实验的论文

| 论文 | 正式发表 | 为什么暂不选作 HotpotQA 主基线 |
|---|---|---|
| [MaFeRw](https://ojs.aaai.org/index.php/AAAI/article/view/34732) | AAAI 2025，A 类会议 | 针对对话改写；T5 初始化、奖励模型和 PPO 是核心。主要数据为 QReCC、TopiOCQA，并非 HotpotQA 原生单题任务。 |
| [Think Then Rewrite](https://ojs.aaai.org/index.php/AAAI/article/view/38527) | AAAI 2026，A 类会议 | 法律/医学领域检索的训练型改写，非 HotpotQA RAG QA 实验。不能只套提示词冒充其 RL 方法。 |
| [Dialogue-RAG](https://aclanthology.org/2025.acl-long.1191/) | ACL 2025 主会长文 | 很相关，但重点是对话中的省略/指代恢复与专用改写模型；我们的首轮单题多跳任务不匹配。 |

MaFeRw 的[代码](https://github.com/TAP-LLM/MaFeRw)与 TTR 的[代码](https://github.com/LIANG-star177/TTR)已定向检查，尚未找到可确认的成品改写器下载。不是说不能做，而是需要另设训练与任务迁移实验。

另外：[DMQR-RAG](https://arxiv.org/abs/2411.13154)很匹配查询改写兴趣，但本轮只核实预印本及 ICLR 投稿，未核实正式录用，不列入严格顶会主基线；也不据此宣称它被拒。ReFormeR、ReflectiveRAG、RaFe 等仍有阅读价值，但不把其短文、Industry、Findings 身份包装成这里要求的 A 类主会长文。

## 精简阅读顺序

1. **S2G-RAG**：全文重点精读，判断是否愿意承担判断器训练门槛。
2. **DualRAG**：重点精读方法、实验与实际生效提示词，作为免微调系统对照候选。
3. **[IRCoT](https://aclanthology.org/2023.acl-long.557/)**：DualRAG 引用并对比的基础方法；读懂为什么推理和检索交替，而不是一次任意重写。ACL 2023，作为基础阅读，不再作为“够新”的主推荐。
4. **[Query Rewriting / RRR](https://aclanthology.org/2023.emnlp-main.322/)**：DualRAG 引用的改写基础；理解检索器、改写器与 Reader 的分工，以及训练反馈。EMNLP 2023，不能按这里的 CCF A 标准充数。
5. **[HotpotQA](https://aclanthology.org/D18-1259/)**：数据与评分章节必读；清楚 fullwiki 和 distractor 不是同一个难度。数据集论文不承担“最新方法基线”的角色。

不要求读完所有参考文献才开工，也不把“基线引用了它”理解为必须全部复现。

## 复现实验边界与硬件

- 先冻结论文版本、代码提交、模型、提示词、查询/检索预算与评价题；建立“论文做法—公开代码—我们适配”差异表。
- 原方法协议与我们两轮预算协议分开报告。改为两轮或十篇候选后，是预算受控/受限检索适配，不能直接对比论文的开放域表格分数。
- 同底座比较：BASE、所选论文 FRESH、GrowRAG FRESH、GrowRAG MEMORY。除整系统对照外，应另做同一 FRESH 底座加/不加经验，才能归因历史额外收益。
- 保存每轮 query、证据来源、停止理由、答案、检索次数、输入/输出 token、延迟与失败；不删除负结果、不用测试金标在线路由。
- API 小样本实验不需要先租 GPU。本机约 16GB 内存、8GB RTX4060 Laptop 可用于现有小样本与轻量组件，不能承诺容纳全维基索引和全部服务。
- 若后续本地跑 7B 推理，建议 24GB RTX4090/3090，系统内存 32GB 起、64GB 更宽裕；具体上下文/批量/并发要实测。Qwen 官方 7B BF16 单批约 6K 输入测试显存为 15.38GB，但该值不是我们的整套系统实测。[官方测试](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- S2G 的 3B LoRA 训练是否能在 24GB 完成，须检查序列长度与训练配置后试跑；不把推理显存当训练显存，不现在要求租多卡机器。

## 本轮实际状态

只进行了文献、代码和本机资源的只读核验，并记录筛选结果；没有新付费调用、没有新 HotpotQA 分数、没有训练、没有声称新 baseline 跑通。旧实验保持原样。本轮先确定方法与忠实复现门槛，再进入实施。
