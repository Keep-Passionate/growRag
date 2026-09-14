# HotpotQA

## 我们最需要掌握的 3 点

- 一道题的 question、context、answer、supporting_facts 各是什么。
- bridge 和 comparison 是推理组织方式，不是 PDF 页面结构。
- 答案正确与找到支持证据必须分别评分。

## 原文与发表状态

[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/)，EMNLP 2018，已审稿的 NLP 顶会数据论文。[官方数据字段](https://github.com/hotpotqa/hotpot#json-format)｜[官方评分](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)。

## 用容易理解的话说

一题包含问题、多篇文章标题及句子、标准答案和标注的支持句。桥接题常需先找中间对象，再沿关系找答案；比较题要取得两个对象的属性再比较。训练集中也有较容易、主要为单跳的题，不能把每题都说成严格多跳。

## 精读定位

先读数据构造与题型分析，再读 distractor/fullwiki 设置，最后对照官方 JSON 和答案、证据、联合指标的评分代码。不要把数据集 type/level 金标直接送给真实路由器。

## 对 GrowRAG 有什么用，不能据此说什么

适合起步研究跨问题经验。标注支持句可以帮助发现“答对但证据没补齐”；EM 是规范化完全匹配，F1 衡量答案词重合，不是语义正确性的万能判断。

我们当前小语料句子检索属于诊断设置，不等于完整 fullwiki 榜单。来源、调试和冻结检查必须隔离；检查题答案不能回流入卡。训练集用于积累经验不等于微调大模型参数。

## 关联与读后问题

关联：[[GrowRAG - DecompRC]]、[[GrowRAG - Sufficient Context]]、[[GrowRAG - ExpeL]]。

读后试答：模型答对但只找到一半支持句，应在哪一层记为成功，哪一层仍未知？

## 阅读记录

- 我的摘录与疑问：

整理：2026-09-14。依据 09-11 官方数据及书目核验，评分说明对应官方实现。AI 导读，个人已读状态未登记。


