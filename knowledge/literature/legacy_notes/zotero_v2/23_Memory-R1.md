# Memory-R1：读前导读

## 一句话理解

Memory-R1 把长期记忆系统拆成“负责写记忆的 Memory Manager”和“负责读记忆并答题的 Answer Agent”，再用最终答案是否正确的稀疏奖励，让两个 Agent 分别学会维护和使用记忆。

## 前置知识与已读论文

01 ReAct 开放 Thought 与 Action 的循环；Memory-R1 把写侧动作明确限制为 `ADD / UPDATE / DELETE / NOOP`。18 Search-R1 用最终答案奖励学习何时搜索、搜索什么，Memory-R1 把相同思路迁移到记什么、怎样改记忆、怎样筛选记忆。12 A-MEM 会链接和更新旧记忆，但主要靠提示；09 Self-RAG 在读取侧用 reflection token 筛选证据，Memory-R1 则用 RL 训练 Answer Agent 做 Memory Distillation。它与 03 ExpeL、04 Dynamic Cheatsheet 最大的差别是：本文主要存用户事实和对话事件，不是可迁移的任务经验或自然语言教训。

## 系统怎样运行

写侧先由普通 LLM 从当前对话抽取事实，再用新事实检索相关旧记忆，Memory Manager 根据二者选择操作并生成修改后的内容。ADD 用于真正的新事实；UPDATE 保留原 ID，把补充信息合进旧记忆；DELETE 删除与新事实冲突的旧记忆；NOOP 忽略重复或无关内容。正文称 NOOP，实际提示使用 `NONE / No Change`。图 1 的两只狗案例说明边界：先有 Buddy，后来“又收养 Scout”不是矛盾，正确动作是 UPDATE 为“两只狗”，而不是删掉 Buddy 再新增 Scout。

读侧从两位对话者的记忆库各取 top-30，共 60 条候选。Answer Agent 不直接吞下全部文本，而先输出真正相关的记忆，再基于它们回答，这叫 Memory Distillation。案例中，未训练模型被“登山”等干扰项诱导回答 mountains；训练后先挑出两条 beach 证据，再回答 beach。

## 两个 Agent 如何训练

Memory Manager 把操作类型和修改文本共同视为动作。动作执行后形成新记忆库，再由冻结的 Answer Agent 回答与当前轮次相关的问题，最终答案 Exact Match 作为 Manager 奖励。冻结 Answer Agent 是为了减少归因混乱，使奖励变化主要来自记忆操作。随后固定训练好的记忆库，训练 Answer Agent 从 60 条候选中生成“所选记忆＋答案”，奖励仍只有最终 EM。两者都比较 PPO 与 GRPO，是分阶段训练，不是联合端到端更新。

## 关键实验

LOCOMO 每段对话平均约 600 轮、2.6 万 token。论文只用第一个对话的 152 个问题训练、第二个对话的 81 个问题验证，其余八个对话、1307 个问题测试，骨干为 LLaMA-3.1-8B-Instruct 与 Qwen2.5-7B-Instruct。以 LLaMA 为例，Mem0 总体 F1/BLEU-1/Judge 为 30.41/22.22/45.68，Memory-R1-GRPO 达到 45.02/37.51/62.74。单独训练 Manager 从 26.73/20.54/47.82 升到 33.05/24.91/59.91；单独训练 Answer Agent 到 37.54/30.64/52.87；加入 Memory Distillation 后从 40.95/34.37/60.14 升到 45.02/37.51/62.74。GRPO 初期更快，最终与 PPO 奖励接近。

论文存在数字书写问题：按表 1，F1、BLEU-1、Judge 相对提升约为 48%、69%、37%，但正文把前两者写反；4.3 节也多次互换 F1/BLEU 数字。引用必须以表格和摘要为准。

## 局限与不可照搬处

“只用 152 个 QA”不等于完全没有额外监督：附录显示 GPT-4o-mini 根据此前 50 轮构建临时记忆库并生成训练状态，RL 状态也不只 152 个。Manager 奖励来自与当前事实关联的 QA，不是未来陌生任务，尚未证明记忆有跨任务长期价值。EM 只验证答案，不验证操作或改写文本是否忠实，错误更新也可能碰巧答对；论文没有报告四类操作准确率、操作分布、长期完整性和误写累积。事实抽取器仍是普通 LLM。

DELETE 尤其危险：从“喜欢 A”变成“不喜欢 A”时，旧偏好仍是历史时间问题的正确证据，物理删除会丢历史。系统也没有版本、出处、置信度和回滚。实验仅在 LOCOMO 对话事实记忆上完成，对 GrowRAG 的跨任务经验迁移没有直接验证。

## 对 GrowRAG 的帮助

最值得借鉴的是分离写侧与读侧。GrowRAG 可以设置 Experience Manager，负责 `ADD / MERGE_AS_NEW_VERSION / DEPRECATE / NOOP`；设置 Experience Reader，从候选教训中筛选适用于当前问题的经验。写侧训练时冻结 Reader 可降低奖励归因混乱，读侧再固定经验库训练何时用经验。不要物理 DELETE，应保留原始轨迹、旧版本和可回滚关系。

经验奖励不能只看产生经验的原题，而应看它对后续未见题的净增益：同一道新题分别在无经验和有经验条件下运行，以正确率或验证分数提升，减去 token 成本、冲突和误导风险，作为价值。实验应分别报告写侧操作准确率、事实忠实度、冲突率、版本恢复率、记忆污染，以及读侧候选 Recall@k、所选 Precision、最终任务增益和上下文成本。

## 精读时追问与关键页

最终 EM 如何同时给操作类型和修改文本分配信用？152 个 QA 实际生成多少 Manager 状态？冻结 Answer Agent 很弱时，正确记忆是否仍获零奖励？DELETE 为什么不保留历史版本？为何不报告 ADD/UPDATE/DELETE/NOOP 的准确率？关键页为正式 ACL PDF 第 2—8 页（方法与结果）、第 12—18 页（案例、训练数据和完整提示）、第 20 页（算法）；正文和表格冲突时以表格为准。
