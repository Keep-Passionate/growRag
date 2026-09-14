# RAQG-QPP

## 我们最需要掌握的 3 点

- 历史查询变体可以帮助预测新查询质量。
- 检索到的变体与新生成的变体，在管线中作用不同。
- “历史经验＋QPP”已经不是空白组合，需要更具体的研究问题。

## 原文与发表状态

[RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval-Augmented Query Generation](https://doi.org/10.1145/3815198)，ACM Transactions on Information Systems（TOIS）2026，已审稿的信息检索领域重要期刊。[作者原文](https://arxiv.org/html/2604.27244v1)。

## 用容易理解的话说

不只孤立看当前查询，而是借助相关查询及生成变体增强质量预测。阅读时分清“从历史中找查询”和“从文档库找证据”，二者都可能叫 retrieval，但成本与信息权限不同。

## 精读定位

读 retrieved query variants、retrieval-augmented query generation 与预测模块，再读跨集合评价和消融。逐个列出每个模块输入中是否含真实检索结果。

## 对 GrowRAG 有什么用，不能据此说什么

它是低成本选择路线的重要排重文献。我们不能只说把记忆引入 QPP 就有新意，而应具体寻找条件信号：例如当前对象绑定、比较属性或禁止变动的约束，是否减少误选？

这里的条件只是候选假设，需要同源动作、等信息预算的实验。也不能不读特征依赖，就把整套 RAQG-QPP 称为完全检索前预测器。

## 关联与读后问题

关联：[[GrowRAG - QPP Query Variant Selection]]、[[GrowRAG - ReFormeR]]、[[GrowRAG - RRM]]。

读后试答：历史查询提供了什么信息，是相似词汇、实际效果，还是明确的适用边界？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；沿用 09-11 导读及 09-08 出版核验。AI 导读，不表示个人已读。


