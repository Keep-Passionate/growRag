# S2G-RAG

## 我们最需要掌握的 3 点

- 先表达缺什么实体、属性、关系或证据，再定向修复。
- 累计证据应对应真实来源句子，不是大模型自由补写事实。
- 区分“证据足够所以停”与“预算用完只好停”。

## 原文与发表状态

[S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](https://aclanthology.org/2026.acl-long.1185/)，ACL 2026 主会长文，已审稿的 NLP 顶会论文。[作者代码](https://github.com/nianaaa/S2G-RAG)。

## 用容易理解的话说

充分性/缺口判断器读取原问题和累积证据，输出 `sufficient` 与 `gap_items`。证据不足时按缺口构造下一条查询，再提取相关证据；足够或到达预算上限后回答。gap 是当前题状态，不天然就是跨题经验。

## 精读定位

读方法流程、judge 的训练数据和结构化输出、证据抽取，再看附录中 sufficiency/gap 提示及轮数限制。对照作者代码找证据句子 ID 怎样映射回原文。

## 对 GrowRAG 有什么用，不能据此说什么

适合作为离线强修复教师或后续 POST 对照。只借输出格式、用冻结 API 提示替代训练过的 judge，必须称“受启发实现”，不能称完整复现。

我们的在线单次 RAG 路线可以不运行循环，把循环留在适配阶段产生经验或标签；这会失去根据现场新证据纠错的能力，需要实测取舍。结构缺口也不能覆盖所有语义改写。

## 关联与读后问题

关联：[[GrowRAG - Sufficient Context]]、[[GrowRAG - ReflectiveRAG]]、[[GrowRAG - RRM]]。

读后试答：历史卡上的缺口，是否在当前题中真的观察到了？如果没观察到，能否照搬？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；依据既有正式论文与作者代码核验。AI 导读，不表示个人已读。


