# QPP Query Variant Selection

## 我们最需要掌握的 3 点

- Query variant 就是同一需求的不同查询文本。
- 预测查询质量与执行完整 RAG 评估查询质量，成本不同。
- 分清检索前预测与检索后预测，不能把已花掉的检索成本忽略。

## 原文与发表状态

[Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines](https://doi.org/10.1145/3805712.3808571)，SIGIR 2026，已审稿的信息检索顶会论文。[作者原文](https://arxiv.org/html/2604.22661v1)。

## 用容易理解的话说

同一问题可产生多种改写。QPP（Query Performance Prediction，查询表现预测）用较便宜的信号估计哪个更值得执行。部分预测器只看查询或预先统计，另一些需要查询已检索出的文档。后者可能省回答成本，却不能说省掉了全部候选检索。

## 精读定位

读查询变体来源、QPP 特征和选择规则，再读检索指标与答案指标的对应实验。看 oracle 时记住：它是事后知道结果的比较上限，不是线上免费的判断器。

## 对 GrowRAG 有什么用，不能据此说什么

后续候选可以是 BASE、FRESH 和历史卡生成的查询。要比较同一候选池，不把“多生成候选”误算成记忆优势。必须记录生成、选择、检索、重排和回答各自的费用与耗时。

预测检索好，不等于预测答案好，也不等于历史相对 FRESH 有额外收益。已有 QPP 选变体研究；我们要验证经验条件是否提供了它缺少的信号。

## 关联与读后问题

关联：[[GrowRAG - ReFormeR]]、[[GrowRAG - RAQG-QPP]]、[[GrowRAG - ReformIR]]。

读后试答：哪些特征在第一次文档检索前真实可得？哪些只是事后拿到的信息？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14；沿用 09-11 导读及 09-05 DOI 核验。AI 导读，不表示个人已读。


