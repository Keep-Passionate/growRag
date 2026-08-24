# 当前导入包

本目录保留当前精读包，并新增方案 B 的 adaptive/agentic routing 与安全经验迁移增量 RDF。新包可能与 08-14 / 08-19 包共享论文；导入后请按 DOI 或 arXiv ID 合并重复条目。

## 证据闭环、REUSE 与 Query Transformation（最新）

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
