# DocLayNet

## 我们最需要掌握的 3 点

- 它是页面布局标注数据，不是现成自然语言问答集。
- 区分单页可见信息与整本文档统计。
- 自建 MetaQA 必须定义哪些字段可输入、哪些只是金标。

## 原文与发表状态

[DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation](https://doi.org/10.1145/3534678.3539043)，KDD 2022，已审稿的数据挖掘顶会论文。[官方 Extra JSON](https://github.com/DS4SD/DocLayNet#extra-json-files)。作者预印本/仓库也使用 Analysis 标题，正式出版题名采用 Segmentation。

## 用容易理解的话说

数据提供页面上的布局框及类别，也有文本和坐标等辅助信息。可以据此构造表格数量、区域关系等问题，但这些问答是我们派生的新数据，不是官方已有 QA。

## 精读定位

读 11 类布局标签、官方切分、页面资产，再看 Extra JSON 的 page_no、num_pages、原文件名及文本单元字段。确认是否拥有完整文档，不能由抽样页猜总页数。

## 对 GrowRAG 有什么用，不能据此说什么

适合后续 DocLayNet-MetaQA 的来源审计。如果任务答案来自布局框，所有方法都应有相同可用输入，不能只给我们人工标注框。需要特别检查完整性、重复页面、文档级切分与标签泄漏。

它与 Hotpot 的“结构”不同：一个是页面布局，一个是跨事实推理。因此不应把两类实验混为同一能力结论。

## 关联与读后问题

关联：[[GrowRAG - DocBench]]、[[GrowRAG - HotpotQA]]。

读后试答：当页面不完整时，模型究竟有资格回答整本统计，还是应该标为不可回答？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；沿用 09-11 出版 DOI 与官方数据核验。AI 导读，不表示个人已读。


