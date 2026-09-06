# 当前导入包

本目录保留历史四日期批次精读包和后续增量。旧包可能共享论文；导入后请按 DOI 或 arXiv ID 合并重复条目。

## 09-05 重新评估（当前入口）

- 新增 `GrowRAG_重新评估_两篇关键近邻_2026-09-05.rdf`，仅 2 条新论文、1 个集合；中文导读在 Zotero 的 Extra 字段，没有重复打包原 86 篇。
- 两篇为 *When Continual Learning Moves to Memory* 与 *LivingRAG*；本轮均只核实为预印本，不冒称已同行评审。
- 旧四批发表状态已逐篇重新核对：`knowledge/literature/2026-09-05_四批阅读包发表状态复核.md`。原 5 RDF 共 106 条、86 篇；加本增量后为 6 RDF、108 条、88 篇。后一个统计不应混入“四批”结果。
- 当前阅读顺序统一见 `knowledge/literature/2026-09-05_必读路线与RAG术语学习单.md`。下方各批优先级是历史记录，不再各自构成一套同时必须完成的阅读清单。
- 新 RDF 保留作者、日期、arXiv ID/DOI、状态与导读，未下载 PDF 附件，也未自动修改 Zotero 数据库。已有条目的最新发表信息请参考复核表，不因旧 Extra 中的标签忽略 ICML/ACL 正式记录。

## 证据闭环、REUSE 与 Query Transformation（08-24 历史包）

- `GrowRAG_证据闭环_REUSE与QueryTransformation_2026-08-24.rdf`：25 条论文、4 个集合；已排除与此前当前目录 RDF 的同名重复项。
- 推荐先读：Sufficient Context → HALT → How Memory Management Impacts LLM Agents → AIR → Self-RAG（旧包回看）→ S2G-RAG（旧包回看）。
- Query Transformation 查重链：DMQR-RAG → SAGE → Think Then Rewrite → ReFeed → MaFeRw → AdaQR → RetPO → SELF-multi-RAG。
- 停止与迭代对照：ReflectiveRAG（旧包）→ SIM-RAG → TASR → Stop-RAG → When Should Multi-Round RAG Stop。
- 每条 Zotero `Extra` 已标注发表状态、做了什么、没做什么以及 GrowRAG 要学习的内容。
- 旧阅读包的逐篇发表层级审计见 `knowledge/literature/2026-08-24_现有Zotero阅读包_发表层级审计.md`。

## 方案 B 自适应路由与安全经验迁移

- `GrowRAG_方案B_自适应路由与安全经验迁移_2026-08-22.rdf`：11 条论文、3 个集合。
- P0 先读：SPARKLE → C-3PO → SegMem-RAG → ReMe → Experience-RAG Skill → ERSkill。
- 算子补读：Query2doc → HyDE → Skill-RAG → State-Aware RAG。
- `RAGRouter-Bench` 暂列 P2；完成单一文本 BASE 后再用于跨 Naive/Graph/Hybrid/Iterative RAG 的外部验证。
- 每条 Zotero `Extra` 已写明发表状态、做了什么、没做什么、GrowRAG 应学什么。带“仅 arXiv”标签的条目截至 2026-08-22 未核实正式接收。

## 已覆盖方向与研究缺口审计包

- `GrowRAG_已覆盖方向与研究缺口_2026-08-19.rdf`：21 条论文、6 个集合，用于查重与定位剩余研究切口。
- 其中 20 条可能已存在于 08-14 / 08-19 包；导入后按 DOI 或 arXiv ID 合并。潜空间版 `ExpWeaver: LLM Agents Learn from Experience via Latent RAG` 是本包新增记录。
- 两篇不同论文都把方法命名为 `ExpWeaver`，不要仅按方法名合并：`arXiv:2605.07164` 是“经验作为可选资源”，`arXiv:2606.01041` / ICML 2026 是“潜空间检索与融合”。
- 建议把本包先导入临时 collection，完成 Zotero 的“重复条目”合并后再移动到正式精读库。
