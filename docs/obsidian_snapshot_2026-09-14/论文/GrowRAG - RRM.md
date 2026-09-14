# RRM

## 我们最需要掌握的 3 点

- OQR 只修当前题；历史经验另提供检索方向与约束。
- 适用条件、证据要求和失败模式比题面相似度更具体。
- 它已比较本题修复与加入历史后的效果，是创新排重必读。

## 原文与发表状态

[RRM: Experience-Driven Reflective Retrieval Memory for Long-Horizon Multimodal Reasoning](https://arxiv.org/html/2607.28156v1)，2026 年预印本；目前仅确认 arXiv 公开版本，不等于未投稿或被拒。

## 用容易理解的话说

先发现空证据、重复查询或连续无新证据等异常。本题在线修复 OQR 只读取当前问题、查询历史和返回证据；历史选择器再看经验条件，提供程序性检索建议。历史答案和原任务事实不进入新题回答上下文。

维护使用反馈和时间衰减。成功与失败经验分开处理，失败经验要求更严格的适配。这不是只找一道文字相似的旧题。

## 精读定位

重点读 Reflective Retrieval Control 下的 Online Query Reflection 与 Triggered Query-Only Experience Reuse；再读 Online Memory Lifecycle Management 和 Table 2 消融。

## 对 GrowRAG 有什么用，不能据此说什么

它是适用条件、query-only 复用和生命周期的直接近邻。不能声称首次提出这些机制，也不能声称首次把 REUSE 与无历史现场修复比较。

待验证差异在于：固定查询候选和执行动作时，少量检索前可观察条件能否改善选择？RRM 的视频多模态任务与 HotpotQA 不同；抽取某机制作对照应标“RRM 启发版”，不是完整复现。

## 关联与读后问题

关联：[[GrowRAG - ReFormeR]]、[[GrowRAG - QCR]]、[[GrowRAG - ReflectiveRAG]]。

读后试答：一个条件需要看检索结果才能确认，能否用于我们的首次文档检索前选择？

## 阅读记录

- 我的摘录与疑问：

整理与关键段落核对：2026-09-14。AI 导读；只确认所列公开版本，个人已读状态未登记。


