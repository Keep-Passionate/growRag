# DecompRC

## 我们最需要掌握的 3 点

- 多跳问题可拆成较简单的子问题。
- 桥接、交集和比较结构需要不同拆分操作。
- 生成候选分解以后还要重评分，不是只贴题型标签。

## 原文与发表状态

[Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://aclanthology.org/P19-1613/)，ACL 2019，已审稿的 NLP 顶会论文。[作者代码](https://github.com/shmsw25/DecompRC)。

## 用容易理解的话说

把复杂问题分成可由单跳模块处理的部分，再组合答案。不同分解可能有效也可能失误，因此还需要重评分。在作者实现中可看到 bridging、intersection、comparison 与 one-hop 等候选路径。

## 精读定位

读问题分解、答案组合和 rescoring 模块；再对照代码中不同分解类别。不要把 Hotpot 的 bridge/comparison 字段当成已经提供了完整分解计划。

## 对 GrowRAG 有什么用，不能据此说什么

可以提供无历史的静态动作模式对照，检验动态历史是否超过合理手工/固定模式库。分解本身可能产生多条检索查询，不能与一次 paraphrase 隐藏成本差异。

按题型设计操作早已有工作。我们的潜在切口是在同类题内部区分适用条件，例如比较年龄和比较任职时长不能盲目使用同一字段。这需要数据，而不是给卡片加字段就算贡献。

## 关联与读后问题

关联：[[GrowRAG - HotpotQA]]、[[GrowRAG - ReFormeR]]、[[GrowRAG - S2G-RAG]]。

读后试答：一个拆分是否丢掉了原问题的时间、否定或对象绑定条件？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；沿用 09-11 官方书目、摘要与作者仓库核验。AI 导读，不表示个人已读。


