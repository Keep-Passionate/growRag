# DG-RAG 的题型识别，如何迁移到经验选择研究？

日期：2026-09-11。性质：本地代码与官方来源核查＋待验证建议。没有重新运行 DG-RAG 或 GrowRAG 实验。

## 1. 结论

可以识别有限的问题类型，并设计不同的改写动作；但“识别类型→针对性改写”本身已有研究。更有价值的问题是：同一大题型内，哪条经验的实际前提符合当前题，哪些不应套用？

还要区分：检索表达不合适（术语、别名、关系描述），与底座缺少必要信息（全局图表清单、页码映射、完整页面）。仅换措辞不能保证恢复不存在于可访问输入中的信息。

首轮继续 HotpotQA 跨题表示实验；DocBench / DocLayNet-MetaQA 留作后续能力边界诊断，不现在替换主实验。

## 2. 本地 DG-RAG 实际做了什么

核查文件：[迁移仓库说明](D:/project/dg-rag-transfer/README.md)、[归档实验口径](D:/project/dg-rag-transfer/docs/experiment-status.md)、[适配器](D:/project/dg-rag-transfer/dg_augmenter.py:67)、[核心操作](D:/project/RAG-Anything-main/reproduce/dg_core.py:565)、[规则语法](D:/project/RAG-Anything-main/reproduce/dg_core.py:593)、[parse入口](D:/project/RAG-Anything-main/reproduce/dg_core.py:952)。

| 操作 | 含义 | 例子 |
|---|---|---|
| Count | 计数 | 多少页、多少张表 |
| Locate | 定位 | 某内容在哪一页 |
| Extract | 提取特定内容 | 指定位置或引用的内容 |
| Lookup | 查元数据字段 | 文档标题等 |

目前主要是英文正则文法，不是任意问题都能理解的训练型分类器。DG 建立文档模型并执行操作，把实际得到的证据附加到问题；无可用证据则原样返回。不是只让 LLM 改写一句查询。

版本边界：`DG_QUERY_GATE=v2` 才启用 v2 eligibility / 留出校准相关门控；默认保留旧 availability-only 行为。不能因为仓库有严格代码，就认为所有历史结果都用了它。旧文档“零回归”表述不是跨任务保证。

## 3. 不只识别题型，还要识别证据需求

以下是待验证建议，不是已实现或已证明的新方法：

| 要识别什么 | 例子 | 用途 |
|---|---|---|
| 需要的操作 | 比较年龄、找关系、计数、定位 | 决定动作家族 |
| 需要的证据范围 | 单属性、双方属性、章节、全文 | 排除局部证据不足以回答全局的问题 |
| 当前底座能力 | 只有文本top-k；有页码；有全页清单 | 判断动作能否执行、是否需要结构工具 |

先用少数明确枚举，只记录影响当前动作的条件；未知单列，不补造几十个环境变量。

- “第2节中的Table X在哪里”：若保留章节名、表名，可以试标题/别名改写。
- “全文多少张表”：需要全局覆盖。只有几块文本时应标记能力不足，不能反复换措辞假装会数全。
- “印刷第4页有什么”：物理页可能不同，需要页码映射。

后续可试两个条件字段：`required_evidence_scope`（证据范围）、`required_backend_capability`（底座能力）。它们是候选，不是默认塞进所有卡；新增字段本身不成立创新。

## 4. HotpotQA 的结构和题型

官方结构如下：[JSON格式](https://github.com/hotpotqa/hotpot#json-format)。

| 字段 | 内容 | 我们何时可用 |
|---|---|---|
| `_id` | 问题标识 | 记账、去重 |
| `question` | 用户问题 | 选择与执行 |
| `context` | 多个段落，各含标题和句子列表 | 检索器建索引；PRE选择器不直接看整份上下文 |
| `answer` | 人工答案 | 预测冻结后的评价；隐藏test没有 |
| `supporting_facts` | 标题与句子编号列表 | 预测冻结后的证据评价；隐藏test没有 |
| `type` | bridge或comparison | 离线分层分析，不入路由；test不提供 |
| `level` | easy/medium/hard | 离线分析，不入路由；test不提供 |

### 正式 type 只有两类

1. bridge（桥接）：先找中间对象再查它。示意“电影A的导演出生在哪个国家”：A→导演→出生国家。
2. comparison（比较）：分别找双方属性，再比较或判断。示意“A和B谁出生更早”或“A和B是否同属一个国家”。

以上是教学示意，不是从数据复制的具体题目。原论文另分析链式关系、条件交集、桥接后属性推断等，它们不是全量JSON自带的更多type。答案是人名、地点或yes/no，也是另一分析角度；操作、答案形式、难度不能混为一类。[原论文](https://arxiv.org/html/1809.09600)

“结构性”有两种含义：

- 推理结构：关系链、两对象比较、条件组合。Hotpot有，适合研究同一大类内哪个动作适用。
- 文档结构：页数、目录、章节层级、版面、图表清点。Hotpot不是为它设计的。

Hotpot以多跳问答为目标，但不代表每题严格需要多跳。原论文train-easy主要是单跳题；另有利用单段线索的捷径。[ACL2019诊断研究](https://aclanthology.org/P19-1416/)讨论了这一点。不能把小样本的单跳比例外推到全量。

### 检索设置与划分

- distractor：两个gold段落＋八个干扰段落。适合小规模句检索诊断；知道候选中含答案不等于推理时可读gold标签。
- fullwiki：从作者维基语料检索，初始检索段落不保证含gold。题内十段检索不能称全库结果。
- 官方有train、dev、隐藏gold的test。项目采用train v1.1，约9万题；dev 7,405题，fullwiki test 7,405题。原论文最初训练量与v1.1不同，执行时记录文件、版本和校验值。

来源：[官方主页](https://hotpotqa.github.io/)、[数据设置说明](https://github.com/hotpotqa/hotpot#data-download-and-preprocessing)。我们用train内部独立角色建经验和开发诊断，之后冻结经验用官方dev评价；不是先拿所有test训练再测。

## 5. “识别类型”已有直接邻居

- DecompRC（ACL2019）把复杂问答拆成子问题，再重评分选择；作者实现含bridging、intersection、comparison、one-hop。因此按结构设计分解不是空白。[论文](https://aclanthology.org/P19-1613/)、[代码](https://github.com/shmsw25/DecompRC)。
- Adaptive-RAG（NAACL2024）用小分类器按复杂度在无检索、单次和迭代检索之间选策略，标签结合策略表现等信息。题面复杂不等于某策略值得，不能只把复杂问题送入复杂RAG就视为新意。[正式论文](https://aclanthology.org/2024.naacl-long.389/)。

最小切口仍是固定候选与能力时，哪些经验信息减少误选；不先宣称胜过完整前人。静态题型规则是必须认真比较的低成本对照。

## 6. 结构数据的后续安排

DocBench公开229个PDF、1,102道题，类别为text-only、multimodal、meta-data、unanswerable；元数据问涉及页数、词数、作者等。[作者仓库](https://github.com/Anni-Zou/DocBench)。正式发表为KnowledgeNLP2025 workshop，不是ACL主会，也不是仅有预印本。[正式记录](https://aclanthology.org/2025.knowledgenlp-1.29/)

DocLayNet本体是KDD2022布局标注数据，不是原生QA。官方Extra JSON含原文档num_pages、original_filename、page_no。系统是否能读取它们必须声明；不能隐藏必要信息后把不可观测问题算作改写失败。[官方元数据](https://github.com/DS4SD/DocLayNet#extra-json-files)

本地[MetaQA构建器](D:/project/dg-rag-transfer/doclaynet/build_doclaynet_metaqa.py:199)把选定COCO split页面按doc_name分组、page_no排序后重组PDF；页数gold为len(pages)，元素数来自这些页的人工标注。没有核验原始PDF所有页是否齐全。

因此当前任务是“重组文档的全局计数”，不保证原始完整文档计数。只有页、表、图、公式、章节标题五个固定Count模板，全部type=meta-data；默认省略元素零计数题。人工counts/框类别不是推理输入。

后续公平要求：

1. 按原doc_name分组，防止同文档题泄漏；DG开发/写稿已用315题不能冒充新盲测。
2. 加事先冻结的释义表达，不能只随机划五个原模板。
3. 单列零数量题，对齐parser与gold对图/公式/标题的口径。
4. 同底座比较BASE、FRESH、静态规则、经验选择；同索引、工具、权限。
5. 若加DG全页算子，所有组都获得相同能力；新增结构访问收益不归因于卡表示。
6. 若静态规则已足够好，接受此任务未必需要历史记忆，不强行制造必要性。

## 7. 阶段关系

Hotpot首轮测跨题表示能否帮助选动作；结构扩展测不同操作/证据范围下能否判断适用边界。共享研究问题，但不混成一个未注明差异的总分。

详见[最小配对实验](../experiments/2026-09-11_经验表示选择_最小配对实验计划.md)。DG灵感保留，DocumentSession仍后置；本轮未改变用户已选Hotpot优先顺序。
