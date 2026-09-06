# GrowRAG 精选阅读包：先读懂，再选路线

整理日期：2026-09-05。当前阶段：阅读与路线讨论；不推进运行代码，不假定用户已经读完任何一篇。

2026-09-06 补充：用户认可“经验选择”和“紧凑表示”两个方向都值得做。原 20 篇顺序不变，只另附 [EPR 与 CEIL 两篇正式会议论文的补充 RDF](GrowRAG_选择与表示_补充2篇_2026-09-06.rdf)，含独立中文导读；EPR 接在 ReFormeR/QPP 后补读，CEIL 等一次多条经验时再读。两者均非本项目方法已优于前人的证明。

## 1. 这次只导入这一份

[下载/打开精选 20 篇 RDF](GrowRAG_逐句精读与直接近邻_精选20篇_2026-09-05.rdf)

这是从历史推荐文献中精选、去重并重新排序的一份整合包，不是又新增 20 篇。原来分散在不同批次的关键论文已收入，不必为了这次阅读再逐个导入旧增量包。旧包仍保留作历史记录，本目录不计入“历史四批共 86 篇”的计数。

- 12 篇核心精读：方法、实验与关键消融逐句理解。
- 5 篇直接近邻：定向核对机制和重合边界。
- 3 篇查询变换/数据配套：对应环节再深入。
- 每篇一条独立中文导读子笔记；另有一条总览笔记。共 21 条笔记。
- 作者、标题、发表信息、可确认的 DOI、原始链接、阅读序号标签齐全。不修改论文原始标题来排序。
- 不含 PDF 附件，不表示已导入本机 Zotero；原文入口在条目与笔记中。

在 Zotero 中使用“文件 → 导入”，选择 RDF。展开条目可见中文导读。集合是三个阅读分组，顺序请看“00 总览”与每篇笔记标题/标签；Zotero 的题名排序不会自动等同阅读序号。

由于你可能已经导入部分旧论文，新导入可能产生重复。先核对 DOI、arXiv ID、完整标题与作者，再合并重复条目；保留原有 PDF、批注和新导读。不要删除原库，也不要把不同作者的同名 ReflectiveRAG 合并。

## 2. 完整阅读顺序

| 序号 | 论文简称 | 正式发表/预印本状态 | 本次最重要的学习任务 |
|---:|---|---|---|
| 1 | [RAG 原始论文](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | NeurIPS 2020，机器学习领域主要顶级会议 | 建立所有后续论文共用的检索与生成术语；不是我们准备直接复刻的工程架构。 |
| 2 | [Query Rewriting](https://aclanthology.org/2023.emnlp-main.322/) | EMNLP 2023 主会，NLP 主要顶级会议。 | 这是理解我们到底修改什么的首要方法论文。 |
| 3 | [IRCoT](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 Long，NLP 主要顶级会议主会长文。 | 先真正理解本题内逐步检索，才能区分现场修复与跨题复用。 |
| 4 | [Self-RAG](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html) | ICLR 2024，机器学习领域主要顶级会议。 | 澄清相关、支持、效用和是否检索这些不同判断，避免一个笼统的“安全分数”。 |
| 5 | [Sufficient Context](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) | ICLR 2025，机器学习领域主要顶级会议。 | 最直接回答你反复问的“证据够不够”，建议这几天优先精读。 |
| 6 | [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/) | ACL 2026 Long，NLP 主要顶级会议主会长文。 | 与你喜欢的“先明确缺什么，再定向改写”最强相关。 |
| 7 | [ReFormeR](https://doi.org/10.1007/978-3-032-21300-6_30) | ECIR 2026 正式论文集，信息检索领域重要专业会议 | 最接近“从过去有效的 query 改写中抽经验再用”的方法，必须精读以避免重复。 |
| 8 | [QPP Query Variant Selection](https://doi.org/10.1145/3805712.3808571) | SIGIR 2026，信息检索领域主要顶级会议 | 连接查询变换、选择器、事后上界与最终 RAG 评价。 |
| 9 | [RRM](https://arxiv.org/abs/2607.28156) | 截至 2026-09-05 仅核实 arXiv 预印本 | 这是我们必须正面比较的最接近工作之一，即使目前只确认预印本，也不能绕开。 |
| 10 | [ERM / RAG without Forgetting](https://icml.cc/virtual/2026/poster/63200) | ICML 2026，官方 poster 记录已核验，属于机器学习领域主要顶级会议工作 | 你认可的“成功后才持久化”的主要参照，同时要看清它的改动位置。 |
| 11 | [Useful Memories v2](https://arxiv.org/abs/2605.12978) | 截至 2026-09-05 仅核实 arXiv | 直接挑战“越总结、越合并，记忆就越好”的假设，决定我们是否需要经验卡。 |
| 12 | [When Continual Learning Moves to Memory](https://arxiv.org/abs/2604.27003) | 截至 2026-09-05 仅核实 arXiv | 必须读的“立意校正”文章：原来会做的题被记忆改坏，已有直接研究。 |
| 13 | [ReflectiveRAG（Verma 等）](https://aclanthology.org/2026.eacl-industry.27/) | EACL 2026 Industry Track，正式审稿会议论文 | 定向读停止与本题内修复即可，避免与 S2G 重复铺开。 |
| 14 | [ReMe](https://aclanthology.org/2026.findings-acl.829/) | Findings of ACL 2026，正式审稿发表 | 比较经验提炼、合并与效用更新；避免重复做一个普通生命周期管理器。 |
| 15 | [GAM-RAG](https://icml.cc/virtual/2026/poster/65843) | ICML 2026，官方 poster 记录已核验 | 了解置信度和不确定性如何共同影响更新，而不是先放一个任意衰减系数。 |
| 16 | [SegMem-RAG](https://ojs.aaai.org/index.php/AAAI/article/view/40320) | AAAI 2026 Technical Track，人工智能领域主要顶级会议正式论文。 | 核对三类记忆和经验路由已有覆盖，防止将层数当创新。 |
| 17 | [LivingRAG](https://arxiv.org/abs/2608.25960) | 2026-08-26 公开，至 2026-09-05 仅核实 arXiv | 最新直接近邻之一：已经有冻结 Graph RAG 部件上的经验增强，必须核对可插拔说法。 |
| 18 | [HyDE](https://aclanthology.org/2023.acl-long.99/) | ACL 2023 Long，NLP 主要顶级会议主会长文。 | 回答你提出的查询变换形式问题；在 Query Rewriting 之后对照读。 |
| 19 | [Query2doc](https://aclanthology.org/2023.emnlp-main.585/) | EMNLP 2023 主会，NLP 主要顶级会议。 | 用一个具体算法理解 expansion，而不是把所有改写都看成同义表达。 |
| 20 | [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) | TACL 2022（Transactions of the Association for Computational Linguistics），计算语言学领域重要顶级期刊。 | 本包最重要的数据设计精读，也是正式期刊论文。 |

01–12 是核心顺序；13–17 属于直接近邻，不要求这几天全部逐句读；18–20 按需补读。其中 MuSiQue 在开始数据实验前仍需精读其构建和证据设置。

每篇独立笔记均包含：为什么读、做了什么、对我们有什么边界约束、重点精读位置、术语、两道自检问题、一页笔记任务、与候选切口的联系、原始来源。导读属于 AI 辅助整理，不替代原文，也不代表你已认可其结论。

## 3. 这几天的短队列

如果 RAG 基础与 Query Rewriting 已经掌握，优先：

**Sufficient Context → S2G-RAG → ReFormeR → RRM → ERM → Useful Memories v2 → When Continual Learning Moves to Memory → QPP。**

这是主队列的“已有基础版”，不是新增阅读要求。前两篇解决证据够不够；中间三篇解决已有查询/记忆方法是什么；后两篇反思经验伤害和表示；QPP 把候选选择、离线上界和评价串起来。

不要求三天读完八篇。读到哪里就讨论哪里；宁可把一篇方法和实验讲清楚，不必赶数量。术语卡壳时看[术语学习单](../../knowledge/literature/2026-09-05_必读路线与RAG术语学习单.md)。

如果尚未读懂基础，仍从完整顺序 01–06 开始，不跳过“原问题”和“检索表达”、“充分性”和“正确性”的区别。

## 4. 哪些是顶会、顶刊，哪些不是？

本包共 **15 篇正式会议论文、1 篇正式期刊论文、4 篇仅确认预印本**，不是“除了几篇顶刊，其余都是 arXiv”。

- 主要顶会工作：RAG（NeurIPS）、Self-RAG/Sufficient Context（ICLR）、Query Rewriting/Query2doc（EMNLP）、IRCoT/HyDE/S2G（ACL）、QPP（SIGIR）、ERM/GAM-RAG（ICML）、SegMem-RAG（AAAI）。
- 须单独标注的正式工作：ReFormeR 是 ECIR；ReflectiveRAG 是 EACL Industry；ReMe 是 Findings of ACL。三者均不应混写成上述会议的主会长文。
- 期刊：MuSiQue，TACL 2022，计算语言学领域重要顶级期刊。
- 当前只确认预印本：RRM、Useful Memories、When Continual Learning Moves to Memory、LivingRAG。未确认正式发表不等于未投稿、被拒或质量低。它们与立意非常接近，不能因 venue 未确认而不读。

发表状态核验截至 2026-09-05，依据会议、出版社、ACL Anthology 与 arXiv 原始记录。ERM/GAM 的正式身份以 ICML 官方记录为准。没有核验当前引用量，所以不虚构引用数、不做“高引排名”。阅读优先级取决于相关性与基础价值，不将 venue 当单篇可靠性的保证。

## 5. 阅读后再选的三个问题

详见[三个候选切口](../../knowledge/decisions/2026-09-05_阅读后待选择的三个研究切口.md)。它们不是旧路线 A/B/C，也不是已经确认的新贡献。

| 候选 | 一句话 | 为什么值得先测 | 难点 |
|---|---|---|---|
| 1. 什么时候值得用历史 | 比较同一状态下历史指导与不看历史的现场改写 FRESH | 最直接检验记忆到底是否增加价值 | 有收益不等于能提前选出来；选择成本可能吃掉收益 |
| 2. 一条经验到底存什么 | 比较具体 q→q′、抽象卡、实例＋必要前提 | 最贴近你对“短小但不丢信息”的关心 | 普通字段堆叠/提示词消融不足以支撑贡献 |
| 3. 少标注时如何谨慎增长 | 少量真实反馈校准无 gold 的代理信号 | 贴近真实使用时没有标准答案的问题 | 代理判断会误判，不能保证自己证明自己正确 |

暂时推荐先测第 1 项，第 2 项作为表示对照，第 3 项留后续；这是建议，不是替用户决定。不要同时扩大成三个完整系统。我们需要用强近邻和相同预算证明具体改进，不能保证不存在相似工作。

## 6. 已授权的单次跟进

用户明确选择“三天后自动继续一次”。已在本对话设置 **2026-09-08 22:50（Asia/Shanghai，北京时间）** 的一次性跟进，自动化标识为 `growrag`。

届时读取本项目记忆与用户新增读书笔记（如有），复核最新近邻，再更新这三个候选问题的比较供用户选择。不假定已读完，不改运行代码、不安装依赖、不运行付费模型实验、不推送 Git，也不自动敲定路线。调度运行本身使用账户额度；本轮已告知并获选择。只有这一轮自动跟进，不持续每日运行。

## 7. 文件格式与核验边界

采用 [Zotero 官方 RDF translator](https://github.com/zotero/translators/blob/master/Zotero%20RDF.js) 的字段与子笔记关系：URL、DOI、作者顺序、标签、集合和子笔记分别保存。导读是笔记，不冒充论文摘要。

结构校验结果见同目录 `VALIDATION.md`。XML/引用关系校验不等于已经在你的 Zotero 界面实际导入；本次不直接修改 Zotero 数据库。
