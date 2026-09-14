# DocBench

## 我们最需要掌握的 3 点

- 区分文本、图文、元数据和不可回答问题。
- 结构信息若在解析时丢失，仅改 query 无法凭空恢复。
- 文档任务的切分应防止同一文档泄漏到来源与测试两侧。

## 原文与发表状态

[DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems](https://aclanthology.org/2025.knowledgenlp-1.29/)，KnowledgeNLP 2025 workshop，正式发表；不是 ACL 主会或顶刊。[作者数据与代码](https://github.com/Anni-Zou/DocBench)。

## 用容易理解的话说

它评估系统读取 PDF 文档回答问题的能力。问题不只有文本事实，也可能涉及图表、页数等元数据，或文档确实没有答案。作者仓库介绍 229 个文档、1102 个问题。

## 精读定位

读四类问题定义、PDF 输入、金标答案和评估流程，再挑几个 meta-data 例子手工确认需要的输入。不要把问题类别金标直接当作路由器预测能力。

## 对 GrowRAG 有什么用，不能据此说什么

适合以后检验 DG-RAG 启发的结构性分支。必须给所有对照相同文档解析和工具权限；否则收益可能来自新增页码/图表工具，而不是记忆。

它不自动提供可靠的长期经验标签，也不适合直接证明跨文档迁移。当前 Hotpot 起步不需要因此重新加入文档记忆层；将文档层作为独立后续问题即可。

## 关联与读后问题

关联：[[GrowRAG - DocLayNet]]、[[GrowRAG - HotpotQA]]、[[GrowRAG - Adaptive-RAG]]。

读后试答：这道结构题是 query 不好，还是检索底座根本没有保存答案所需字段？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；沿用 09-11 ACL 官方 workshop 核验及作者仓库。AI 导读，不表示个人已读。


