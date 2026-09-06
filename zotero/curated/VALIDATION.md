# 精选阅读包核验记录

日期：2026-09-05。目标文件：`GrowRAG_逐句精读与直接近邻_精选20篇_2026-09-05.rdf`。

## 结构检查：通过

- XML 能正常解析；各笔记 HTML 片段也能作为良构片段解析。
- 论文条目：20；去重后完整标题：20。
- 类型：15 conferencePaper、1 journalArticle、4 preprint。
- 作者缺失：0；URL 缺失：0。
- 独立导读子笔记：20，与论文一对一关联；另有一条总览，共 21。
- 集合：4（一根集合、三个分组）。
- 内部标识唯一；悬空关系：0。
- 文件大小：100501 bytes。
- SHA-256：`06B1795DD703C30F83C345F0F58DEDF4B17E75BCC910185D2555D9E728019625`。

## 元数据与内容依据

- 以项目历史 RDF 的完整作者和标题为基础，结合 2026-09-05 的正式发表复核；不沿用旧 Extra 中已经纠正的创新性断言。
- RRM、ERM、GAM-RAG、Useful Memories、ReFormeR、S2G 六个重点近邻的作者名单与顺序，本轮另以 arXiv、ICML 官方、ACL Anthology、出版社 DOI 元数据再次核对，无需修正。
- 核心来源：`knowledge/literature/2026-09-05_四批阅读包发表状态复核.json`、同日近邻复核、必读路线和原始论文页面。
- Useful Memories 导读明确针对 2026-08-29 更新的 v2。
- 参考 [Zotero 官方 RDF translator](https://github.com/zotero/translators/blob/master/Zotero%20RDF.js) 的字段映射生成 URL、DOI、集合、标签及子笔记关系；中文导读没有混作论文摘要。
- 15 篇会议包括 Findings 与 Industry，README 分别标注；4 篇预印本仅表示未确认正式发表，不表示未投稿。

## 核验边界

这是文献文件的结构与来源核对，不是对所有论文结果的独立复现，也不是对全部 20 篇本轮逐句重读的声明。AI 导读中的候选切口是建议，尚无实验支持其优越性。

本次未在实际 Zotero 界面执行导入，因此不宣称已通过真实客户端导入测试；未修改其数据库，未下载/附加 PDF。用户实际导入后，应展开一篇论文确认导读可见，并谨慎合并旧重复条目。

## 2026-09-06 两篇补充包

`GrowRAG_选择与表示_补充2篇_2026-09-06.rdf`：2 篇 conferencePaper、2 条独立导读子笔记、1 个集合；XML 与笔记 HTML 片段可解析，内部悬空关系为 0。EPR 作者/DOI/页码以 ACL Anthology 为准；CEIL 作者/会议/卷页以 PMLR 官方记录为准，未虚构 DOI。没有执行实际 Zotero 客户端导入，不含 PDF。

SHA-256：`D638E2F1ED964C3E4733C491FF6D5A94BF5F853A06E27F8C18F3314149530DCA`。原 20 篇 RDF 未变。
