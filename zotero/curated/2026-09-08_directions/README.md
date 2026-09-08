# 2026-09-08：四个研究方向的阅读包

先读[方向比较与创新边界](../../../knowledge/decisions/2026-09-08_记忆AgenticRAG研究方向_QPP与阅读路线.md)，再按方向阅读，不要求一次精读 30 篇。

## 导入哪一份

- 已有之前的论文：优先导入 [本轮新增 8 篇](GrowRAG_本轮新增8篇_2026-09-08.rdf)。
- 想从一份文件获得四个方向的完整分组与新导读：导入 [整合精选 30 篇](GrowRAG_四个研究方向_精选30篇_2026-09-08.rdf)。其中 22 篇此前已推荐，不是又新增 30 篇。

两份二选一，不需要同时导入。在 Zotero 用“文件 → 导入”选择 RDF；展开论文可见独立中文导读子笔记。完整包有一根集合和四个方向子集合；同一篇进入多个方向时共享条目。题名保持正式标题，阅读顺序使用标签和总览笔记，不篡改题名。

已经有旧条目时，导入可能产生重复；核对 DOI、完整标题与作者后再合并，保留旧 PDF/批注和新笔记。不要删除原库，不要将不同作者的同名工作合并。此包不直接修改 Zotero 数据库，不含 PDF 附件。

## 先读的短队列

已有基础：**QPP Query Variant Selection → ReformIR → RAQG-QPP → ReFormeR → RRM → QCR**。

选择长期记忆方向时，再读 **Useful Memories v2 → When Continual Learning Moves to Memory → ERM**；选择修复闭环时，再读 **Sufficient Context → S2G-RAG**。

基础不熟时先补 Query Rewriting、ReAct、Self-RAG。若读过旧文，不必重读全部，按[逐篇导读](逐篇导读.md)复核对应方法与消融即可；不会替你自动标记已读。

## 新增八篇分别补什么

| 论文 | 身份 | 作用 |
|---|---|---|
| ReformIR / When More Reformulations Hurt | SIGIR 2026 主会完整研究论文 | 多候选、预算与漂移已有强近邻，必须先读 |
| RAQG-QPP | ACM TOIS 2026 正式期刊 | 历史查询＋候选生成＋QPP，必须先读 |
| Adaptive Query Performance Prediction for RAG | ACM TOIS 2026 正式期刊 | 检索后预测与自适应深度；选读 |
| Q2EI | Findings of ACL 2026 | 查询也可以压缩，而不只是扩写；选读 |
| SMR / From Token to Action | Findings of EMNLP 2025 | REFINE/RERANK/STOP 与漂移诊断；选读 |
| WeWrite / When & How to Write | SIGIR 2026 Industry | 何时拒绝改写、如何避免历史偏好干扰；选读 |
| RECAP | Findings of EACL 2026 | 用户意图本身不明确或变化时的理解；跨领域选读 |
| Am I on the Right Track? | IR-RAG@SIGIR 2025 workshop；CEUR 2026 出版 | QPP 与 Agentic RAG 各轮搜索的分析桥梁；选读，非主会 |

TOIS 是信息检索领域重要顶级期刊。完整包统计：21 篇正式会议论文（包括 Findings/Industry）、1 篇正式 workshop、2 篇期刊、6 篇仅确认预印本。预印本不等于没投稿，也不等于低质量；未核验当前引用量，因此不虚构“高引”数字。

## 核验范围

- `metadata_base.json`：22 篇已有元数据快照，保留真正核验日期，纠正旧 QCR 类型等问题，不沿用旧 Extra 的夸大表述。
- `metadata_new.json`：8 篇新增，7 篇从出版社 Crossref 登记获取作者、正式题名和 DOI；workshop 对照 CEUR 正式卷目录。活动年与出版年分开。
- `reading_notes.json`：30 篇中文导读来源；包括做了什么、学什么、已有重合、没据此证明什么及阅读问题。
- `validation.json`：实际结构校验结果及 SHA-256。完整包 30 条论文、31 条笔记、5 个集合；增量包 8 条论文、9 条笔记、4 个集合。作者和 URL 不缺失，内部标识唯一，无悬空关联，笔记 HTML 可解析。
- 参照 [Zotero 官方 RDF translator](https://github.com/zotero/translators/blob/master/Zotero%20RDF.js) 的 URL、DOI、作者顺序、集合与子笔记映射；没有在实际 Zotero 客户端执行导入测试，不把 XML 校验说成真实导入成功。

RAG-QPP（TOIS）本轮仅核出版社摘要与作者仓库，不声称已精读全文。其他论文也是定向章节核验，不是本轮全部 30 篇逐字重读或独立复现。AI 导读供阅读定位，不替代原文。

本目录生成脚本只处理本地公开文献元数据与导读，不联网、不读取密钥、不调用模型；重新生成不会改变旧阅读包。
