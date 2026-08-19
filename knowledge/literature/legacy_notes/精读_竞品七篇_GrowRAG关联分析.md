# GrowRAG 竞品精读笔记：七篇核心相关工作
> 整理日期：2026-08-05 | 用途：GrowRAG 论文差异化定位 | 来源：Claude Code 综合分析

---

## 速查表：七篇工作的系统定位

| 缩写 | 第一作者 | 年份/会议 | 操作对象 | 训练 | 对 GrowRAG 的意义 |
|------|---------|---------|---------|------|-----------------|
| ReflectiveRAG | Verma (Amazon) | EACL 2026 Industry | per-query 修复策略 | ❌ | **单题修复基线 BR** |
| ERM | Hu (Emory) | arXiv 2602 | 文档 key 嵌入 | ❌ | 最直接竞品，但改的是索引 |
| GAM-RAG | Wang (Fudan) | arXiv 2603 | 句子级记忆向量 | ❌ | raw+EMA 被覆盖，降为消融 |
| ExpWeaver | Zhao | arXiv 2605 | 经验调用时机 | RL(可选) | 门控器想法已被覆盖 |
| MemCon | Jiang (UCLA) | arXiv 2607 | 记忆操作策略(MDP) | ✅(bandit) | 在线控制已被覆盖 |
| Memory-R1 | Yan | ACL 2026 | ADD/UPDATE/DELETE | ✅(RL) | 端到端记忆 Agent 竞品 |
| Amber | Qin | ACL 2025 Findings | 多 Agent 迭代记忆 | ❌(主体) | 多 Agent RAG 路线太重 |

---

## 1. ReflectiveRAG
**【单题检索修复的最强基线 — 我们摊销的对象】**

### 基本信息
- **标题**：ReflectiveRAG: Rethinking Adaptivity in Retrieval-Augmented Generation
- **作者**：Akshay Verma, Swapnil Gupta, Siddharth Pillai, Prateek Sircar, Deepak Gupta（Amazon）
- **发表**：EACL 2026 Industry Track，Rabat，Morocco，March 2026
- **链接**：https://aclanthology.org/2026.eacl-industry.27/
- **DOI**：10.18653/v1/2026.eacl-industry.27
- **Zotero 文件**：`D:\program\zotero\storage\DUGV4YEY\`

### 核心机制
两个模块：
1. **SRR（Self-Reflective Retrieval）**：用 1.5B SLM（无需微调 generator）迭代判断检索证据是否充分；不充分则改写 query 重新检索
2. **NR（Noise Removal）**：基于嵌入的对比过滤，删除重复/无关 passage；-31% 证据冗余

### 实验结果
| 数据集 | ΔEM | ΔF1 | 额外延迟 |
|--------|-----|-----|---------|
| HotpotQA (distractor) | +2.7 pp | +2.5 pp | +18 ms |
| WebQuestions | 优于 DeepRAG | — | — |
| 内部 QA（50M CC distractors） | 优于 DeepRAG | — | — |

**1.5B SLM 打败 7B RL 模型**：证明不需要重型训练，反思推理已经足够。

### GrowRAG 启示
- 这是我们的 **BR（单题修复基线）**：每道题都重新反思，有效但代价高
- 我们的贡献是：让这些修复在后续相似查询中**被安全摊销**，并在检测到干扰时**退化回 BR**
- 论文中应明确将 ReflectiveRAG 列为 BR，我们的"额外成本"减少目标相对于它来衡量

---

## 2. ERM（Evolving Retrieval Memory）
**【最直接竞品 — 但改文档索引，我们不改】**

### 基本信息
- **标题**：RAG without Forgetting: Continual Query-Infused Key Memory
- **作者**：Yuntong Hu\*, Sha Li\*, Naren Ramakrishnan, Liang Zhao（Emory University / Virginia Tech）
- **发表**：arXiv 2602.05152（2026-02-06）
- **链接**：https://arxiv.org/abs/2602.05152
- **Zotero 文件**：`D:\program\zotero\storage\8H3G28TQ\`

### 核心机制
将**query-time query expansion → 持久化为 document key 更新**（index-side），三步：
1. **Correctness-Gated Feedback**：只有当检索+生成都正确时才触发更新（防止噪声写入）
2. **Selective Attribution**：每个 expansion unit 只更新它真正提升了的那些文档 key
3. **Progressive Key Evolution**：softmax 归一化的批量 key 累积更新，有停止判据（patience + saturation）

理论保证：query/key 扩展等价性证明；key 序列收敛证明；amortized cost = O(T^{1/α})（Zipf 分布下次线性）。

### 实验结果
- 数据集：BEIR + BRIGHT，共13个域
- BM25 + ERM：平均 +46% nDCG；密集检索 +13–15%
- 推理-密集任务最大收益：AoPS +2200%（BM25），LeetCode +44%（BGE-Large）
- **零推理额外开销**（key 已预更新，inference 和原始检索同速）
- 下游 QA 提升：BM25 +6 pp，密集检索 +2–4 pp

### GrowRAG 差异化
| 维度 | ERM | GrowRAG |
|------|-----|---------|
| 存储位置 | 文档 key 嵌入（**索引内**）| 独立经验账本（**索引外**） |
| 存储内容 | query expansion deltas | query rewrite / evidence_ptr / answer |
| 是否修改索引 | ✅ 是 | ❌ 否（plug-and-play）|
| Abstention | ❌ 无（key 总是被更新）| ✅ NO_MEMORY 是一等动作 |
| 语料漂移应对 | 需要重新 adaptation | 版本 hash + TTL 自动失效 |

---

## 3. GAM-RAG（Gain-Adaptive Memory RAG）
**【积累检索经验+自适应增益更新 — 原方案 raw+EMA 被覆盖，降为消融】**

### 基本信息
- **标题**：GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation
- **作者**：Yifan Wang\*, Mingxuan Jiang\*, Zhihao Sun, Yixin Cao, Yicun Liu, Keyang Chen, Guangnan Ye, Hongfeng Chai（复旦大学）
- **发表**：arXiv 2603.01783（2026-03-03）
- **链接**：https://arxiv.org/abs/2603.01783
- **Zotero 文件**：`D:\program\zotero\storage\3HGLTY4D\`

### 核心机制
三阶段框架：
1. **图构建**：用 spaCy NER 构建轻量三层图（entity→sentence→passage），**无需 LLM 关系提取**（索引成本仅 1.66M token）
2. **记忆引导迭代检索**：entity 激活 → 句子传播（memory 加权）→ passage 聚合 → 多跳扩展
3. **Kalman 启发增益更新**：每条句子记忆维护 `(m_task, m_time, π_task, π_time)`；LLM-as-judge 反馈 → 非对称噪声更新（正反馈 R_pos < 负反馈 R_neg，避免过度抑制 bridge evidence）

**关键公式**：`K_i = π_i / (π_i + R_i)`（自适应学习率）；perplexity π 控制更新幅度，越稳定越保守。

### 实验结果
| 指标 | 结果 |
|------|------|
| vs 最强基线 | +3.95% GPT-Acc（0-turn）|
| 5-turn 记忆 | +8.19% GPT-Acc |
| 推理成本 | -61%（5-turn 后同题延迟减少 58%）|
| 数据集 | 2Wiki / HotpotQA / MuSiQue / TimeQA / Medical |
| 主要基线 | HippoRAG2, LinearRAG, PoG, DyG-RAG, REMINDRAG |

### GrowRAG 启示
- **原方案中 "raw/abstract 双轨 + EMA gain"** 与 GAM-RAG 重叠：GAM-RAG 已经做了 Kalman gain + memory perplexity
- 这两个设计改为第7节消融，**不再作为标题贡献**
- 我们的核心差异：GAM-RAG 需要**图结构**（entity→sentence 图）；我们不需要图，更 plug-and-play
- GAM-RAG 无 abstention（memory 总是参与推理）；我们有 NO_MEMORY 一等动作

---

## 4. ExpWeaver
**【经验调用时机研究 — "何时调用"想法已被覆盖】**

### 基本信息
- **标题**：Rethinking Experience Utilization in Self-Evolving Language Model Agents
- **作者**：Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yanyan Zhao, Yu Zhang, Yang Wu, Dandan Tu, Bing Qin, Ting Liu
- **发表**：arXiv 2605.07164（2026-05-08）
- **链接**：https://arxiv.org/abs/2605.07164
- **Zotero 文件**：`D:\program\zotero\storage\3SZRD3XK\`

### 核心机制
- 现有 agent 使用经验的方式只有两种：**初始化时一次注入** 或 **每步强制注入**，均为静态
- **ExpWeaver**：将经验作为**可选资源**，agent 在推理不确定时主动调用
- 测试：4 个 agent 框架 × 7 个 LLM × 3 类环境（通用任务），consistently 优于固定调用策略
- 结论：**always-on 实际差于 init-only**；RL 可以进一步放大"选择性调用"行为

### GrowRAG 启示
- 证明 NO_MEMORY/REUSE 的三路决策是有意义的（ExpWeaver 独立验证了选择性调用优于强制调用）
- 但 ExpWeaver 研究的是**通用 agent 任务**，不是 RAG-QA 的检索修复场景；我们在 RAG 场景下做更精细的研究
- **我们的工作不把"何时调用"当做主创新，而是把它作为系统设计的一部分**（回应 ExpWeaver 的启示）

---

## 5. MemCon
**【在线学习记忆控制策略 — "训练记忆门控器"已被覆盖】**

### 基本信息
- **标题**：Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents
- **作者**：Eric Hanchen Jiang, Zhi Zhang, Yuchen Wu, Levina Li, Dong Liu, Xiao Liang, Rui Sun, Yubei Li, Edward Sun, Haozheng Luo, Zhaolu Kang, Aylin Caliskan, Kai-Wei Chang, Ying Nian Wu（UCLA 等）
- **发表**：arXiv 2607.13591（2026-07）
- **链接**：https://arxiv.org/abs/2607.13591
- **本地 PDF**：`D:\project\GrowRAG\docs\literature_notes\2607.13591_MemCon.pdf`（已下载）

### 核心机制
- 将记忆操作建模为 **MDP（马尔可夫决策过程）**：状态=当前任务上下文，动作=\{retrieve, inject-plan, consolidate, forget, noop\}
- 学习方法：**轻量 tabular contextual bandit + UCB 探索**（无需预训练，仅需二元任务反馈）
- 包装任何已有记忆后端，不依赖具体 LLM
- 解决的问题：早期任务阶段稀疏检索更好；重复目标可复用 plan；长任务流需要 consolidation + pruning

### 实验结果
- 6 个基准 × 3 个 agent 框架 × 3 个 LLM backbone
- 任务成功率最高 **+15.2 pp**，token 使用 **-5–20%**

### GrowRAG 启示
- MemCon 用 contextual bandit 学习**何时/如何**使用记忆的策略，在**通用 agent 任务**上验证
- **关键区别**：MemCon 研究的是对话/任务型 agent 的记忆操作；我们研究的是 **RAG-QA 中检索经验的存储粒度与信任机制**，场景、问题、存储内容完全不同
- MemCon 的 bandit 需要学习（即便轻量），**我们第一版完全不训练**

---

## 6. Memory-R1
**【端到端可学习记忆 Agent — 若做此路线则正面竞争且训练成本高】**

### 基本信息
- **标题**：Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
- **作者**：Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schuetze, Volker Tresp, Yunpu Ma
- **发表**：ACL 2026 Long Papers
- **链接**：https://aclanthology.org/2026.acl-long.583/
- **DOI**：10.18653/v1/2026.acl-long.583
- **Zotero 文件**：`D:\program\zotero\storage\V2QBRVGI\`（2508.19828.pdf）

### 核心机制
- 双 agent 架构：**Memory Manager**（决策 ADD/UPDATE/DELETE/NOOP）+ **Answer Agent**（利用记忆回答）
- 训练：**RL（PPO/GRPO）**，仅用 **152 个 QA pair** 即可收敛（outcome-driven reward）
- Memory Manager 学习对话历史上的记忆操作序列
- 结论：RL > SFT > heuristic

### 实验结果
- LoCoMo / MSC / LongMemEval 等长期对话记忆基准
- 3B 到 14B 模型规模下均优于 baseline

### GrowRAG 启示
- Memory-R1 研究的是**对话历史记忆**（ADD/UPDATE/DELETE 的是对话中的事实/关系）
- 我们研究的是**RAG 检索经验的跨查询摊销**：问题性质、存储内容、训练开销完全不同
- **不建议走端到端 RL 路线**：(1) 训练成本高；(2) 场景与 Memory-R1 有重叠会被审稿人比较；(3) 无训练版本已足够强

---

## 7. Amber（Qin et al.）
**【多 Agent 迭代记忆 RAG — 路线太重，不适合作为第一篇】**

### 基本信息
- **标题**：Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation
- **作者**：Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, Xiaoman Liu, Xianwei Meng
- **发表**：ACL 2025 Findings
- **链接**：https://aclanthology.org/2025.findings-acl.418/
- **DOI**：10.18653/v1/2025.findings-acl.418
- **Zotero 文件**：`D:\program\zotero\storage\U48W2HX3\`

### 核心机制
三个子模块协作：
1. **Agent-based Memory Updater**：多 agent 协作更新记忆库
2. **Adaptive Information Collector**：自适应信息收集，含查询改写和停止判断
3. **Multi-granular Content Filter**：多粒度内容过滤

整体是**强耦合多 agent 系统**，每次迭代都需要多 agent 通信。

### GrowRAG 启示
- Amber 已经覆盖"多 Agent + 迭代记忆 + 查询改写 + 停止判断"，这条路线在 2025 年已经被占领
- 系统复杂度高、可复现性差、难以做 plug-and-play 消融
- **结论：GrowRAG 第一篇应走轻量单 Agent 旁路路线，与 Amber 的多 Agent 强耦合路线明确区分**

---

## 差异化总结对比表

| 维度 | ReflectiveRAG (BR) | ERM | GAM-RAG | ExpWeaver | MemCon | Memory-R1 | Amber | **GrowRAG（我们）** |
|------|-------------------|-----|---------|-----------|--------|-----------|-------|-------------------|
| 存储对象 | 无（stateless）| doc key 嵌入 | 句子记忆向量 | agent 经验 | 记忆操作策略 | 对话历史记忆 | 多 agent 记忆库 | **检索修复增量** |
| 修改索引 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | **❌** |
| 需要图 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | **❌** |
| 需要训练 | ❌ | ❌ | ❌ | RL 可选 | ✅ bandit | ✅ RL | ❌ | **❌（第一版）** |
| Abstention | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | **✅（一等动作）** |
| 主场景 | 单题修复 | 检索索引 | multi-hop QA | 通用 agent | 通用 agent | 对话记忆 | RAG 强耦合 | **RAG-QA 流式** |
| 数据集 | HotpotQA | BEIR/BRIGHT | 2Wiki/Hotpot | ALFWorld 等 | 6 agent 基准 | LoCoMo | QA 数据 | **HotpotQA/2Wiki/TriviaQA** |

> **我们的核心差异化**：GrowRAG 是第一个在 RAG-QA 场景下，系统研究"检索修复经验的存储粒度（raw vs abstract）× 信任机制（none/once/EMA）× 触发条件"对下游QA准确率影响的工作，且完全无训练、不碰索引、不需图结构。
