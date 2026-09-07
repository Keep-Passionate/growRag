# GrowRAG 当前项目记忆

更新日期：2026-09-07
状态：0.5.0新增可编辑逻辑图与有限问题模式候选匹配，本地695测试通过；本轮只跑零API规则诊断，不自动可信REUSE。此前0.4.0真实首批仍为32源1卡、16目标无卡，BASE6/16、FRESH5/16、REUSE未执行；145次API约0.0209324元，无新模型成绩。0.4.1已补推送成功。QPP门控、训练、文档层/多轮后置，创新待证；9月8日文献跟进不扩权为付费实验。
权威性：当前入口；旧路线 A 决策保留为历史记录，但不再代表当前主线

## 2026-09-07 最新：独立构思文档与经验增量归因

- 用户本轮要求整理独立 MD/PPT，沿 DG-RAG 外置启发→ReAct 循环→经验表示/录入/路由→缺口修复→有限停止的逻辑串联历史文献；本轮是文档与图，不扩展运行代码、不读密钥、不新增收费实验。
- 主文 `docs/GrowRAG_项目构思与文献脉络_2026-09-07.md`，历史完整索引 `docs/GrowRAG_构思文档_历史文献索引_2026-09-07.md`；两张可编辑图在 `figures/2026-09-07_project_concept/`。图是规划，不是全部实现。
- 用户建议的 REUSE−FRESH 被明确为主配对：FRESH−BASE 是现场修复收益，REUSE−BASE 是整套流程收益。必须同 PRE/POST 决策起点、同动作预算、分支隔离、gold 后置、库冻结；差值是有噪声的条件性观察，不等于每个字段的因果证明。
- 文献核验补充：RRM 已有 OQR 与 OQR＋经验消融；ExpWeaver（Rethinking…2605.07164）已有调用位置的经验移除对照，不能把有无经验配对当首创。该篇仅确认预印本，与 ICML2026 的 ExpWeaver Latent RAG（2606.01041）不是同一篇。ReMe 也已有验证、适配和效用淘汰。
- QPP 的 QSD(Pre) 已用历史近邻查询效果预测；S2G 本来就是轻量可插拔并有结构化 gap。创新仍需落在条件化的 REUSE−FRESH 价值估计及最少前提表示，并胜过直接近邻；不是系统层数或模块拼装。
- ReflectiveRAG 的 Sim/Conf 停止式实现与附录算法存在复现疑点；借鉴收益停止动机，不声称复现完整停止算法。搜索预算结束不等于证据充分；S2G 到轮数上限调用 reasoner 不等于自动弃答。
- 新文纠正旧回滚措辞：可以撤销无依据查询约束，但不能为了让证据充分而隐藏真实冲突。检索证据的来源有效性与支持冲突必须保留。
- DG-RAG 仅按用户描述作为工程启发，未虚构正式论文身份。QueryEpisode/ExperienceCard 为首版，DocumentSession 继续后置。当前数据仍沿09-06正式协议的官方划分＋train内约80/10/10，不复活旧比例。
- 提出“按条件记收益”的最小候选机制供后续验证，不将其升级为已获用户选择的算法或已证明创新；现有少量模型结果仍未证明经验额外收益。
- 交付验证：86个原审计标题全部保留、另有16条补充与4条未确认线索；图已本地渲染并目视检查，正文/索引/入口合计107个本地链接有效。独立审阅后补清硬阻断与软适用反例、预定义配对分母和覆盖率、建库/在线成本分账。业务代码未改，因此未把此前695测试写成本轮新测试。

## 2026-09-07 最新：用户要简单图并继续开发

- 用户继续授权本地编码/测试与既有分支同步；本轮不新增收费实验或读取密钥。实现与图入口 `docs/pattern_matching_v1.md`，图源 `figures/2026-09-07_pattern_routing/`。
- 新0.5.0 `experience/pattern_matcher.py`：有限两姓名年龄比较语法识别，不以来源实体重合排名；UNKNOWN→BASE，匹配仅CANDIDATE，不触发真实REUSE或晋升卡片。姓名样式、实体类型、完整自然语言前提均未语义验证，不冒称可信controller。
- ReviewedScope为开发代理暂定的显式范围（review_origin=developer_provisional），非用户/人工认证；绑定全CardMemoryView SHA和matcher_version，任何压缩改版或规则变更使旧声明失效，需重新检查。首版只声明age，不把来源卡自动泛化到height。
- 明确拒绝同源ID/同源规范化题、缺范围、版本不符、非PRE、非paraphrase、需要当前证据、否定/多对象/混合属性/不支持语法。旧0.4.0真实入口和卡片schema不改。
- `experiments/pattern_probe.py`真实运行的是本地规则：9个合成边界题3候选/6BASE；旧16开发题0候选/16BASE。不是LLM实验、无新答案分数、无gold评价；数据组分开、原冻结库不变。输出 `runs/2026-09-07_pattern_probe_v1/report.json`。
- 下一步先来源操作归因/范围标注及独立匹配诊断，再冻结新批次做BASE/FRESH/REUSE比较。手写规则只是脚手架，不当作论文创新；小模型、QPP门控、EMA、多轮、文档层后置。
- 上轮同步障碍已解除：普通push已将a79d923/fad5cb1推送到GitHub并核对远端fad5cb1；CI34073693240已启动，后续API查询遇限流，未确认终态，不绕过限流。新0.5.0验证/提交状态待本条后追加。
- 0.5.0全量695项测试通过（新增62）、Ruff和格式检查通过；独立只读审查未发现高优先缺陷。运行前后旧batch_result和frozen_library文件SHA完全一致，API日志仍145份。PNG/SVG已实际渲染检查，避免中文溢出，并用虚线标明尚未接通的自动真实路由。
- 0.5.0功能提交 `6c5628afed46cea9425b826c726297f5b5e57a6b` 已正常推送并用git远端引用核对；main仍 `404107a`。干净导出695测试、Ruff、CLI通过，验证副本在TEMP的 `growrag-export-audit-v050-7dbf801252b440ed8d7deadc865c1d7f`。GitHub API限流期间未继续查询CI、不新建已通过CI的版本标签；远端CI终态仍待核对，不能把本地验证说成云端通过。

## 2026-09-06 真实完成／09-07 继续离线排查

- 完整结果：`knowledge/experiments/2026-09-06_PRE真实建库与比较_结果.md`；原始产物在 `runs/2026-09-06_pre_pilot_32_16_v1/`。不覆盖原始文件。
- 实验commit `8081af4`，运行前本地与干净导出598测试通过；GitHub CI 34043676389 Python3.11/3.12通过。分支已推送，main未合并。
- 145次API全部返回固定Qwen快照，输入73606、输出7764 tokens，总估算0.0209324元，无失败或预算阻断。约5元是本次整批上限，未用余额不自动授权下一批。
- 32来源只产生1张年龄比较候选；16目标全无卡，BASE答对6/16、FRESH5/16，REUSE根本未执行，不报0/16。两实体共同属性／关系交集题不能硬套数值比较卡；首先是覆盖不足，不单独证明相似度匹配失败。
- 唯一源题从仅有Mary出生日期到同时检出Kim出生日期，支持召回0.5→1、弃答→Mary Pierce。卡仍candidate、matched_trials=0；两个支持文档不算两个独立经验。
- 免费本地复算：完整成功query与仅保留两个人名的query检出同一有序top4，因而未证明“vs＋比较属性”是有效成分；年龄→身高适用外推也未验证。不能把成功来源的模型自我概括当因果解释。
- 负例：Wenling/Xinzheng问题只增加located也导致关键支持1→0、正确→弃答。语义保真与检索有益应分开；这是FRESH负例，不是REUSE误用。
- 后续先做不收费的查询成分检索诊断，再做模式覆盖、条件匹配和独立新题迁移。先不训练大模型、不启EMA/文档层/多轮，不为刷出正结果修改这16题。
- 原报告“无卡未完成”与“无预算阻断=未知”的文案随后修正并补测试；计分和收费原始JSON不回写。组合RAG token未知不用于推算总费，总费用以API账本为准。
- 0.4.1后续已实现并跑完免费 `experiments/retrieval_probe.py`：归档BASE/FRESH有序证据回放一致；仅实体、实体＋age、实体＋vs均与完整FRESH同序top4，而只删tennis短语仍只召回0.5支持。输出在 `runs/2026-09-07_pre_source_retrieval_probe_v1/probe.json`。人工事后诊断、无Reader/API，不给新query虚构答案成绩。本地633测试与静态/格式检查通过；真实0.4.0标签已推送，后续代码独立留痕。
- **同步状态（09-07 00:12北京时间）**：0.4.1代码已在本地提交 `a79d923e7eb1f6ccefd20c0bb0d24b6e79909af2`，干净导出633测试/Ruff/CLI通过。三次不强推的Git同步遇连接重置或443不可达；GitHub API复核远端开发分支仍为 `8081af4`。因此0.4.1尚未推送、尚无其远端CI，不宣称完成；下一轮先正常推送现有分支并查CI，不重跑收费实验。0.4.0真实运行版本及其标签已在远端，main未改。

## 2026-09-06 最新确认与开发：首批总预算5元已授权

- 用户明确同意并要求继续持续开发；授权整个新批次源题建库＋提卡＋目标对照累计估算5元，不是每阶段/每分支5元，不是无限期经费。
- 新0.4.0代码入口 `docs/pre_pilot_v1.md`：PreSourceRecord存独立BASE/FRESH；PreSourceProvenance明确区别旧Episode来源，ExperienceCard外壳复用，不冒充BASE-first轨迹。源成功只建candidate，不计目标matched_trials。
- 固定现有Hotpot train200中的32来源＋16内部校准目标，旧163/21/16角色不变；官方dev/test未用，无新数据下载。16题已曝光，算回归开发不算盲测。
- 新入口固定原Qwen快照、temperature=0、关闭思考，768输出/256总请求/5元总估算；官方北京短档价格再次核验。实例独立但只共享一个无对话状态的传输/总预算账本，不分发多份额度。
- 来源录入：真实改query、FRESH答案正确、答案/支持不退化且至少一项增量；不是每张卡必有新增证据。提卡只读原query/实际query，不读gold/答案/文档。自然语言前提尚未独立语义核验。
- 目标前冻库；当前词面Jaccard只作弱候选基线。无卡默认BASE路由，但REUSE保留未执行，不复制BASE成绩。每臂立即落盘，磁盘失败停调用，中断不自动恢复/重发。
- 本条为运行前实现记录；收费试跑是否完成、实际请求/费用与结果须追加实际记录，不把预算授权或本地测试当成已调用。

## 2026-09-06 最新讨论：进入正式开发的范围与确认项

- 用户认为可以正式开发，询问下一步及待确认项。完整建议在 `knowledge/decisions/2026-09-06_正式开发首版_范围与待确认.md`；新建议不自动冒充用户已选定或新 API 授权。
- 核查发现重要障碍：旧 Episode 要求 BASE-first 且 repair 引用前轮 gap；卡视图要求来源 repair，不能诚实承载独立空证据 PRE FRESH 来源。下一开发先补独立来源操作引用/建卡，不伪造顺序，不把旧 POST 经验改 stage 后继承成功。
- 建议顺序：真实 PRE 来源与候选卡 → 新批量入口/共享总预算/失败落盘 → 小批真实三分支 → 最小可绕过选择器。继续复用0.3.0，不重写整套架构。
- 默认建议：先语义改写、无合适经验回 BASE，FRESH 保留研究对照；来源库在目标批次冻结。扩写、在线 reliability/EMA、多轮、文档层和参数训练后置，用户可调整。
- 首批建议最多32来源＋16内部开发目标，都是已有 train 角色；旧校准题已曝光，仅算回归排错，不称新盲测。后续更大内部批次另冻结；官方 dev/test 不用于建库调参。
- 建议新批次累计估算 API 上限5元（建库和全部分支共用，不是每支5元），尚未确认；运行前须复核模型/价/上限。此次仅规划，无业务代码改动、密钥读取、收费调用或Git推送。

## 2026-09-06 最新实施：前置三分支执行与逐题报告

- 入口 `docs/query_comparison_v1.md`。沿用统一外壳、不同正文和原外置 RAG 接口，补上首次检索前的 BASE／FRESH／REUSE 比较，不重写底座。
- 相同原问题、空证据、每支一次完整 RAG；FRESH／REUSE 各至多一次改写。实验先固定 form/intent，再选匹配卡；两支共享 `growrag-paired-query-v1` prompt，仅 REUSE 获得白名单历史视图。
- 所有分支组件先完成配置/来源检查，再执行；实例分开、记录声明的配置指纹。指纹不是第三方实现无共享状态/不漂移的证明；种子只固定分支顺序。
- gold 不进执行器，运行后才评价。保存原始记录、JSON 报告与易读 MD；完整三种差值、正确变错误/弃答、实际是否改 query、失败和未知费用分开。相同 query 的答案波动不归功于改写；不自动晋升经验。
- 独立审计发现普通组件异常可能中断比较，已纳入本轮修复：不保留异常正文，保留可观测的调用费用，不能把未知当 0；配置/策略非法仍应尽早暴露。
- 只支持 PRE_RETRIEVAL 的显式候选卡与单查询语义改写/扩写。POST 共享前缀、多查询拆分、自动适用性选择、线上 judge、可靠度更新均未冒充完成。
- 本轮零 API 演示在 `runs/2026-09-06_typed_query_comparison_mock_v2/`，问题/卡/改写/答案均为手写测试替身，三支打平是预设，不是实验结论。v1 不覆盖；未读取密钥、下载数据或训练。
- 版本提升 0.3.0，保留独立分支 `codex/query-card-actions-v1` 和既有 `v0.2.0-research.1` 回退点；不合并 main，不提交密钥、原始数据和运行日志。完成验证后在本条后补记录。
- 完整本地 497 项测试通过，较上一版新增 73 项；静态和格式检查通过。新增 CI 零 API 三分支演示。普通组件异常/非法返回隔离测试通过，用户中断与显式配置/策略/跨轮证据冲突仍保留中断行为。
- 功能提交 `3b8da81` 已推送；干净源代码副本 497 项通过，GitHub CI 34041393239 的 Python 3.11／3.12 均成功。标签 `v0.3.0-research.1` 随后创建、推送并核对解析值；main 未动。详见 `knowledge/experiments/2026-09-06_前置三分支_版本验证.md`。
- 下一步先固定少量 Hotpot train 内来源/开发题，真实比较新卡相对无历史 FRESH 的额外作用，再做笔记表示与选择器。不应同时扩张多轮、EMA、文档层和新数据集。

## 2026-09-06 最新确认与实施：统一卡外壳、不同动作正文

- 用户明确同意“统一外壳，不同动作正文”，允许继续编码并推送 GitHub，要求可回退的版本控制。最新实现入口为 `docs/query_cards_v1.md`；图在 `figures/2026-09-06_query_cards/`。
- 沿用 ExperienceCard，v1 增加 ParaphraseBody／ExpansionBody；新正文替代旧 slot_template，形式与目的分离，语义改写不要求 gap。拆分仍不执行。
- `experience/query_views.py` 编译实验预选卡的只读视图，核对全部来源 episode／修复步骤；所有来源 ID 和规范化问题指纹均做自源隔离。只白名单字段进入改写，不送历史答案、文档、gold 或验证分数。
- 新视图接入 RewriteDecision→新版固定 prompt→原独立 RAG 外循环。旧 APIRewriter 拒绝 typed card，旧纯文本 prompt 和数据格式保留，不无声更改已有实验。
- 当前只做阶段／必要证据／形式一致性等结构检查，不把自然语言前提传入 prompt 冒称自动适用性判断。候选卡仅用于明确指定的诊断，不视为 trusted；旧 registry 的验证门槛未放宽。
- 版本号更新为 0.2.0；独立分支 `codex/query-card-actions-v1`；`ff1da96` 为上一轮实现回退点。密钥、data/runs/models 不入 Git，不强推、不自动合并主分支。
- 本轮没有新 API 调用或训练；新增代码的 mock 测试验证接口，不是模型效果。下一步仍是固定语义改写形式的 FRESH／REUSE 配对，再研究便宜适用选择器，不同时扩张所有模块。
- 新增 42 项卡片测试，完整 424 项通过，静态检查与格式检查通过；独立审计指出的嵌套 JSON 重复键歧义已经拒绝并补测。
- 功能提交 `4a25352` 已成功推送开发分支，远端 HEAD 核对一致；从该提交导出的干净代码副本也通过 424 项测试。GitHub CI 34039319001 的 Python 3.11／3.12 两个作业均核验成功，随后推送标签 `v0.2.0-research.1`。验证记录：`knowledge/experiments/2026-09-06_统一卡代码_版本验证.md`。

## 2026-09-06 最新实施：前置查询层＋独立 RAG 外循环

- 最新记录：`knowledge/decisions/2026-09-06_外置查询层_动作分流与最小实现.md`；代码图在 `figures/2026-09-06_outer_loop/`。当前用户明确允许最小编码、测试和训练，覆盖下文此前的暂停条件。
- 外层可在首次 RAG 前选择 BASE 或改写，也可在 RAG 返回后继续；不改写原问题、不改检索器／Reader 内部。底座必须能区分原始问题和搜索查询；不暴露证据时只运行一次并标记未验证。
- `outer_loop.py` 是有限循环，不是新 agent 框架。AWM 借鉴的是工作流经验，不能直接等同于显式状态机。默认首次 BASE，继续时 FRESH；训练后的最佳路由尚未实现。
- `query_actions.py` 分开 BASE/FRESH/REUSE、改写形式 form 与目的 intent。允许无结构性 gap 的语义改写／扩写；拆分仅预留并在调用前拒绝。旧卡全量字段迁移、自动条件选择与当前 API 新 prompt 实测未完成。
- 外层累计 Evidence 只供下一步规划；当前 Reader 每轮只读当轮证据，不自动累计阅读。完整 RAG 循环会重复回答生成费用，不能冒称只多一次检索的成本。
- 经验压缩／改动生成新版本 candidate，显式重新提交适用条件和成本，验证次数归零／时间未知，父版本保留。`revised_candidate()` 已实现；不是自动压缩器，不预断置信度一定变低。EMA 后置，调用多不等于可信。
- 已有充分性信号、预算、重复 query、无收益代理等停止边界；judge 出错保留回合与费用。尚无真实充分性 judge 接入新循环；停止不等于正确，也没有最佳答案自动回退。
- QPP 有检索前／后两种；原查询效果、候选改写差值、REUSE 相对 FRESH 收益是不同目标。本轮实际训练的是第一个，不冒称第三个。
- 现有 train 前 200 题角色不变：163 memory_seed 本轮未用于拟合，21 selector_train 拟合 StandardScaler＋Ridge，16 calibration_dev 检查。开发 MAE 0.261906，对照常数 0.270833，仅微小工程诊断，无正式效果证据。结果在 `runs/2026-09-06_preflight_qpp_v1/`，未接路由。
- 完整 382 passed、新代码 Ruff 通过，4 个独立审计问题已修复。新 RAG 流程用测试替身；BM25 标签与 Ridge 拟合真实 CPU 执行，无 GPU、付费 API、密钥读取或新数据下载。
- 模块不要求逐个首创。当前主假设为有限笔记下的适用前提保留与新题相对 FRESH 增量，压缩后的适用边界复验作为相关验证；不能回退为“条件字段／QPP／状态机组合即创新”。

## 2026-09-06 此前约束：先图审，重新收紧创新边界

- 最新记录：`knowledge/decisions/2026-09-06_图先行_官方划分与记忆创新复核.md`；两张图及提示词在 `figures/2026-09-06_design_review/`。用户确认图的输入、箭头和方法细节前，不继续业务编码或实验。
- 图 1 分训练建库与冻结后的新题运行；历史仅指导查询，不直通回答事实。已有配对时点在首次检索之后，不可称为纯检索前 query-only 路由。自动选择与语义支持检查仍为待实现虚线模块。
- 图 2 是“按当前证据局部使用经验步骤”的候选解释，未获方法定稿授权。例子为虚构示意，不能当实验；当前代码仍每支至多一次修复。
- 查重新增强约束：QCR 已有目标条件化笔记、重绑定、适用前提和验证；AMA-Agent 已保留动作依赖并随状态变化弃用失效分支。条件门、部分步骤/状态机并非可直接宣称的新颖点。
- RRM 并非只按相似度；有条件/证据需求/失败模式及新证据后更新。Hindsight Memory-PRM、HiMPO、TreeMem 已涉及反事实或阶段信用；“精细记功”也不能单独当首创。
- DeltaMem-lite 官方缓存标 Anonymous ACL submission，仅核对摘要/首屏，未读完整方法、未确认录用；不能称 ACL 正式论文。ExpWeaver 存在同名论文，本项目指 Zhao 等 arXiv2605.07164 的经验工具调用工作。
- 待选：局部适用查询操作（候选方法）＋相对 FRESH 的支持增量归因（验证机制）；压缩保持适用边界后置。这是研究假设，不是已确认创新；需正面对照 QCR/RRM/AMA 等。
- 数据沿用官方 train/dev/test，不用测试建库。先 Hotpot/2Wiki 各自 train 建库并各自评价；MuSiQue 先冻结 Hotpot 来源迁移。train 内约 80/10/10 是本项目来源/选择器/内部校准用途，不是官方要求；官方 dev 若用于最终公开比较，就不同时用来反复调方法。
- QPP Query Variant Selection 已有 SIGIR 2026 正式记录，后续更新 RDF 状态，不应继续只标预印本。
- 本轮只作图与文档；不读取用户 API 密钥、不新增 RAG 调用、不训练、不推送业务更改。绘图由内置图像工具完成。

## 2026-09-06 此前实施：单步真模型试跑与数据协议

- 实际结果：`knowledge/experiments/2026-09-06_首轮真实Hotpot试跑_结果与下一步.md`。78 次普通百炼 API，无重试/换模型/传输失败；58,338 输入与 4,000 输出 tokens。运行 commit `3d58c399bf552cc3d4b7bc7c159f3e04a2979d1a`，启动时干净；数据和源库有哈希。
- 目标 8 题：BASE 正确/错误/弃答=3/1/4；FRESH=6/0/2；REUSE=6/1/1；后两者平均支持召回均 80.21%，全部逐题 EM/F1/召回打平。只说明本批修复有帮助，不证明经验优于 FRESH，更不证明正式基准优势。
- 核心失败审计：REUSE 有“FRESH 弃答→错误作答”，其引用指向正确人名却回答另一个人；另有 sufficient 时 BASE 错答、修复可救回；修复擅加 2024。引用 ID 检查不等于蕴含，gap 不直接作安全停止器。后两支打平也不能掩盖弃答与错答的区别。
- 目前选择器是无阈值词汇 top-1，虽然卡中存条件但未执行条件门；两张卡覆盖太窄。本批是弱基线诊断，不是完整可信复用系统。下一步先条件门＋允许不用经验和支持审计，再比较 QPP/笔记写法；不得对看过的 8 题调好后宣称新题泛化。
- 用户明确授权架构设计、编码、小开发题测试和 GitHub 同步。架构先记录在 `docs/architecture_v1.md`，当前实现使用独立实验执行器，不由旧 facade 决定研究方向。
- 首版六个职责：数据隔离、本地检索、固定模型、候选经验、配对执行、离线评价。自动 gap、来源笔记提取、BASE/FRESH/REUSE 已接通并完成这次真实运行；完整控制器、QPP、训练等仍未启用。
- 新权威数据协议为 `knowledge/datasets/2026-09-06_正式数据协议与阶段划分.md`，覆盖旧的 40/35/25 与 dev 500 pilot 建议：保留官方划分，train 内约 80/10/10 用于经验来源/选择器训练/校准；本轮不触碰官方 dev。比例由我们设计，不称来自某篇论文。
- 只下载 Hotpot train 前 200 条完整记录：稳定散列划分得到 163/21/16，首轮来源 8＋校准 8 题在调用前冻结。它不是全量随机样本，尚未下载或实体化全部 90,447 条的划分。
- 当前没有梯度训练；“训练”首先指从独立 train 题积累经验。以后选择器训练只能用 train 的独立角色。2Wiki 后续域内验证，MuSiQue 先冻结跨数据迁移，QASPER 文档层后置。
- 仅用现有普通百炼固定快照执行低成本试跑：上限 96 请求/768 输出 tokens/估算 1 元，无重试；不为换平台新增采购。GLM 免费 Flash 可作开发备选，但需新账号密钥与另立模型实验，不混入本轮。
- 来源 FRESH 正确且有新 gold 支持只晋升候选，不叫 trusted。目标阶段冻结源库，词汇相似度选 top-1 是弱基线，不是我们最终可信算法；gold 只用于离线反馈。DocumentSession、EMA、QPP、模型训练暂不激活。

## 2026-09-06 最新：按量 API 连通与 Token Plan 购买边界

- 详情：`knowledge/decisions/2026-09-06_TokenPlan与按量API_首次真实连通检查.md`。Token Plan 个人版只用于允许工具的交互式使用，不能用于我们的批量实验脚本；Lite/Standard/Pro 均非本项目实验采购建议。
- 用户文件 `qwenAPI.md` 最初为空，后保存了普通北京百炼业务空间地址与一个按量 API Key。保留原 endpoint，文件加入 Git 忽略，不显示/提交密钥，不改为订阅或中转接口。
- 已真实发送一次短 JSON 连通探针；请求/返回均为 `qwen3.7-flash-2026-07-15`，usage 为 23 输入/5 输出，未重试。记录 `runs/2026-09-06_qwen_flash_connectivity_v1/summary.json`。这是首次真实 API 调用，但不是 RAG 或改写结果；下文“API=0”属于此前状态。
- 本次按官方北京短上下文价估算 0.0000086 元，不是已核对账单。关闭思考，最多 1 请求/64 输出 tokens。未购买产品、未下载数据、未跑研究评测。
- 新增普通百炼预检查与本地句级 BM25，304 项测试通过。后者只是候选集诊断，不是 fullwiki；真实首检、gap、经验积累及完整配对尚未贯通。
- 后续低成本质量实验使用固定快照；不因用户希望正向结果而筛选好题或丢弃负结果。Hotpot gold 自动评价优先，模型 Judge 作为辅助；对话助手不冒充固定 GPT API 裁判。

## 2026-09-06 此前：双层起步、分层反馈与最小代码已实施

- 用户本轮明确允许在讨论同时实现最小骨架，不再处于纯文档规划阶段。详细记录：`knowledge/decisions/2026-09-06_HotpotQA_双层起步与分层反馈_实施记录.md`；操作入口：`docs/minimal_experiment_quickstart.md`。
- 用户考虑取消文档层，尚未确认永久删除。工程先不依赖 DocumentSession，只让单题状态/轨迹与跨题经验输入参与配对；保留文档层构思/旧接口，QASPER 可在以后检验真实文档内导航价值。
- 采用证据层与答案层反馈分开记录：有新文本≠有新支持≠改善答案；答案正确也不等于当前动作有功。来源有效性与目标 REUSE 相对同状态 FRESH 的增量另分开。非 gold 反馈只能标代理，不能自动成为可信标签。
- 前人并非只看相似度：ERM 已有检索/生成 OR 验证，GAM 判句子证据，RRM 已按适用条件/证据要求/失败模式选择及维护经验。RRM 还明确采用三层；Useful Memories v2 §6.2 已指出抽象会剥离适用前提。层数、条件字段及发现丢前提有害都不能归我们首创。
- 仍探索两个相连问题：低成本地选择真正优于 FRESH 的经验；给真正带来证据/答案增量的操作记功与保留必要信息。必须有具体机制和实证，尚不宣称已具备独创算法或优于近邻。
- 新 `experiments/paired_runner.py` 从外部首检快照执行 BASE/FRESH/REUSE，修复最多再检索一次；禁止 source=target、mock/real 混用、证据越界、超 top_k、覆盖旧结果；失败保留费用与审计链。
- 新 `experiments/hotpot.py` 读取本地官方 JSON、gold 与题型/难度不进 RuntimeQuestion。分层评价实现句级 gold 支持与答案 EM/F1，`answer_supported=None`，不把引用存在当成文本蕴含；并非完整 Hotpot 官方评价替代品。
- 新 `api_client.py` / `llm_adapters.py` 是待 live 验证的 HTTPS 与 query/reader 适配器。默认网络关闭、密钥从显式环境变量读、不碰 Codex 登录信息；禁止失败后 mock 回填。记录请求/响应、模型/版本、usage 与失败，已知密钥回显脱敏；未调用外部模型。
- mock 为手写测试替身，不是费用估计器。演示中 REUSE 的成功是预设剧情，不能当算法结果。已运行 `runs/2026-09-06_single_repair_mock_v1/run.json`，明确 API=0；回放只重读记录不重新执行。
- 本轮 256 项测试通过、新代码 Ruff 通过；使用已有 `.venv` Python 3.11.15，没有安装、下载全库/模型、训练或推送 Git。未实现真实首检/索引、自动 gap Judge、自动建卡、QPP 选择或完整修复循环。分支费用明确未含外部首检成本。
- 本机实测约16GB RAM、RTX4060 Laptop 8GB显存。API-first无需现在申请GPU；后续实验室24GB显存/64GB RAM是8B原精度或14B量化的合理起点，具体估算见 `knowledge/experiments/2026-09-06_资源与API选型_核验.md`。
- 真实试跑还需厂商/地区、固定模型与费用上限。不要在聊天中索取密钥。当前对话输出不等于 GrowRAG API 结果；GPT-5.5/5.6 的官方 API 文档也不等于账户已可调用。

## 2026-09-06 用户已确认：HotpotQA 跨题经验优先

- 用户对优先级问题的明确回复：“先跨题经验：HotpotQA 起步，文档层后置（推荐）”。此项由建议升级为已决定，后续不重复询问同一选择。
- 第一阶段研究跨题查询修复经验，起步数据为 HotpotQA；三层架构保留，DocumentSession 不作为最小配对实验的前置条件。
- 最小实验沿用同一首检快照、一次后续修复、FRESH/REUSE 配对和 BASE 参考；笔记写法与选择策略逐项比较，不同时上线多轮、EMA、自动合并或小模型训练。
- QASPER 仍是文档层后续候选，不因本次选择而视为具体数据协议已全部批准。样本量、模型/设备和真实费用上限仍待运行前确认。
- 本次确认只更新计划状态，没有启动真实模型、下载数据、安装依赖或推送 Git。

## 2026-09-06 本轮答疑：先让比较对象能被理解、被检验

- 最新完整记录：`knowledge/decisions/2026-09-06_表示实验_QPP与三层数据适配_答疑.md`。本轮依学术实验规划流程记录假设和验证边界，没有运行模型。
- 用户不理解“表示实验”，以后先称“经验笔记写法对比”：同一源步骤写成具体改写例子/抽象规则/例子＋最少前提，固定当前题与证据比较。不是 embedding 训练，也不是把 S2G 当跨题卡片系统。
- QPP 可加入。尚未生成的 query 没有该 query 的传统 QPP；先生成 FRESH/REUSE 再做检索前 QPP，是易解释的对照协议，须计两支生成费用。首检的检索后信号只能描述当前状态，不能充当未执行新 query 的检索后信号。
- 更收紧的候选主张：保留经验中必要前提，能否在当前已知/缺失证据下改善相对 FRESH 的选择；必须胜过 QPP-only 与已有 ReFormeR/RRM 风格情境/条件匹配。条件字段、相对收益目标、三层组合本身不新。
- 新近邻：Tian 等 RAG Utility/Answer Quality Prediction 已由 Springer 核准 ECIR 2026；Dado 等 Predicting the Benefit of Retrieval Augmentation 的 v3 作者报告 CIKM 2026，出版端本轮未独立确认。不能再将“预测相对增益”作为首次。
- 重要更正：GAM-RAG 不仅同题反复记忆，Table 2 / 附 D 已有按 2Wiki 题型拆开来源/目标的 Different Query 实验。不能把测独立新题当作它没做过。旧数据总表相关表述已补正。
- 数据建议：HotpotQA 先调通跨题单步修复，2Wiki 关系条件验证，MuSiQue 更困难迁移后置。三者不天然提供同文档连续多题，用户对 DocumentSession 的质疑成立。
- 文档层建议单独用 QASPER（NAACL 2021）诊断：同论文多个问题，人工答案/证据；按文档划分、只从已读证据累计导航/别名/歧义，禁止后题 gold，文档结束冻结。需先审计现象频度；不能保证别名足够多或第二层一定有效。
- 文档层首版只做有出处的别名/术语、章节位置、歧义避坑；局部 action 统计延后。对照无局部记忆、已读段落缓存与同预算静态文档地图，不能把缓存效应包装成经验学习。
- 三层保留设计，但不要求首个实验全部上线；若第二层没有独立增益，不强行宣称三层必要。
- 当前 `external/` 仍只有 README，并非某个上游 fork；`paired.py` 只整理输入分数，不执行配对。下一步是共享首检快照→真实 FRESH/REUSE 分支→隔离 gold 评价。建议成熟检索组件＋薄实验层，非从零写搜索引擎。
- QPP-4-RAG 的 MIT LICENSE 已核；S2G 未核得 LICENSE，ReFormeR 仅核到 README MIT 声明，直接复制前再核。上游 oracle 命名也有歧义，应以实际使用真实结果或预测分数区分。
- 上述答疑产生时新增方案属建议；同日后续用户已确认 HotpotQA 跨题优先，其余具体参数仍待定。未克隆上游、下载全库、安装依赖、付费实验或推送 Git。第一批先共享协议/假模型测试，再接单步真实调用与逐题报告；模型端点/硬件、真实预算运行前确认。

## 2026-09-06 最新确认：两个方向共同推进规划

- 用户已明确认可研究问题 1、2，不应在下一次复盘又要求从三个问题中只能选一个。少 gold 的谨慎增长仍作后续储备。
- 最新入口：`knowledge/decisions/2026-09-06_经验选择与紧凑表示_联合研究方案.md` 和 `knowledge/experiments/2026-09-06_双方向最小实验与编码路线.md`。
- 建议联合立意、实验分开：固定实际 source_step_id 比较具体查询对/抽象卡/实例＋最少前提；再固定表示比较选择器，最后交叉验证。联合主线与实施顺序是建议，尚未由用户选择。
- 一份不可变来源记录、多种读取视图，不复制三套库。修复前状态必须由当时证据支持，不从成功后的文档回填；事后抽象前提标待验证。角色化 query-pair 只绑定当前题可支持的实体。
- 对旧“长期卡默认不读取具体 q→q′”作范围修订：为表示研究允许受控比较具体视图；最终默认仍待实验。完整 CoT、历史答案、gold 与文档正文不进入默认在线经验。
- 首个实验只在同一首检状态后再做一次修复，先不接完整多轮/在线 DocumentSession/EMA。保留成功来源优先、冻结 prompt、单 query episode、三层职责。
- 选择目标是相对 FRESH 的实际增益，而非来源答对。无 gold 的线上只做预测，不预知目标结果；gold 和配对结果只在隔离开发/评价侧使用。
- 两种预算口径分开：同改写/检索机会与同总成本上限。总成本下允许 FRESH 用省下的选择开销做预定的额外生成/自检。增加等次数多 FRESH 对照，避免 oracle 仅因多抽样取最大值占优；最佳标签需要稳定性审计。
- 当前代码重点缺口：旧 facade 不执行 FRESH；默认候选是 Jaccard 而非 BM25；新卡/episode 尚未接通真实管线；配对 baseline 主要受限单轮 BASE；预算是声明而非分支实测。下一次实施优先协议/视图、同状态分支执行器、FRESH 对照报告、再做选择器。
- 候选默认组合：联合立意且先表示实验、桥接实体/关系修复、质量优先。用户本轮要求给选择，尚未确认这三项。
- 只补 EPR（NAACL 2022）与 CEIL（ICML 2023），后者延后选读。效用导向示例选择与去冗余已有工作，不能独立宣称创新。补充 RDF：`zotero/curated/GrowRAG_选择与表示_补充2篇_2026-09-06.rdf`，原 20 篇包不改。
- 本轮是研究与编码方向规划，没有获得付费真实实验预算；没有修改 src/tests、安装依赖、运行模型或推送 Git。

## 2026-09-05 必须先读的修正

- RRM 已直接覆盖 OQR/FRESH、适用条件、query-only 经验与生命周期；ReFormeR 已覆盖成功 query pair 的模式抽取与当前情境选择。可插拔、三层、reliability/applicability 的概念分工都不是独立新贡献。
- Hu 等 *When Continual Learning Moves to Memory* 已报告 baseline-success / baseline-fail 的 retention / new learning 和 harmful reuse。不能再把 correct→wrong 指标本身当新贡献。
- LivingRAG（08-26 新预印本）在固定图 RAG 上增加经验，以 grounding/novelty 控制写入，是上轮之后新增直接近邻。
- “安全”目前是待评估的质量退化控制目标，不是代码保证；现有 2 次观测、0.70 风险上界是 demo 默认值。逐卡严格统计门存在样本稀缺，应先评估全局/动作族校准，卡级记录作为特征。
- 历史四日期批次共 5 RDF、106 条、86 篇：62 正式会议/论文集、1 TACL 期刊、1 作者确认 COLM camera-ready、4 workshop、18 仅确认预印本。ERM、GAM-RAG、潜空间版 ExpWeaver 的 ICML 2026 官方记录已确认。不能把 Findings/Short/Industry 直接等同低质量，也不能把 arXiv 链接等同未录用。
- 建议首先比较相同检索前缀和预算下的 FRESH、具体历史 query-pair REUSE、抽象卡 REUSE；先看 oracle 是否有额外空间，再研究便宜选择器。保留 DocumentSession 设计，暂不让其在线控制成为首个实验前置条件。
- 40%/35%/25% 是官方 train 内工程初值，不是论文标准比例。MuSiQue-Full 不可回答标签仅针对给定缺证据上下文，不能直接用于能搜回缺失证据的开放全库拒答评价。

完整依据：`decisions/2026-09-05_重新评估_项目路线与贡献边界.md`；`literature/2026-09-05_可插拔经验系统与安全复用新颖性复核.md`。下文未修订的细节继续作为 08-24 架构候选，不代表全部应同时实现。

## 当前主问题

> 保留基础检索器、索引与回答模型，在同一个证据缺口下，什么时候复用历史查询修复比不看历史的现场修复更有益？能否用执行前可见信息作出有效、低成本且少退化的选择？

暂定简单标题：

- 中文：RAG 中的查询修复经验选择性复用
- 英文：Selective Reuse of Query Repairs for RAG

以上为 09-05 新标题建议，尚未由用户敲定；得到实证前不在标题暗示安全保证。

## 已确认的架构

1. 基础 RAG 是可替换但在一次实验中冻结的 `BASE`，可以是普通向量/混合 RAG，之后再验证 GraphRAG；经验层不修改其索引、retriever 或 generator。
2. v1 采用 `BASE-first`：原查询先检索一次；证据足够就直接回答，不进入复杂路径。
3. 首检不足时输出结构化 gap，再选择：
   - `REUSE_REPAIR`：有历史上可靠且当前适用的经验；
   - `FRESH_REPAIR`：没有安全可用的历史经验；
   - `STOP_ABSTAIN`：无进展或预算耗尽。
   `CONFLICTED_OR_UNCERTAIN` 进一步分为：有可检索核验目标且仍有进展时 `FRESH_VERIFY_OR_REPAIR`；否则撤销 REUSE 分支并退回干预前检查点，回滚后仍不足则 `STOP_ABSTAIN`。
4. 检索前 `PROACTIVE_REUSE` 只作为后续消融，不是首版默认。
5. 原路线 A 的 reliability/applicability/harm gate 被吸收到方案 B，作为 REUSE 的安全部件；路线 A 不再单独作为论文。
6. 原方案 C 不做主线；充分、无进展和预算停止只是方案 B 的必要控制。

## Agentic / Adaptive 定位

- 标准术语用 **Agentic RAG**。
- 规则和冻结 prompt 阶段：带 Agentic 修复循环的 adaptive/modular RAG。
- 控制器能够按状态选择 `ANSWER / REUSE / FRESH / STOP` 后：轻量级单控制器 Agentic RAG。
- 多 Agent、插件式、不修改底座、按简单/复杂路由都不是创新。
- 路由目标不是表面“问题简单不简单”，而是“离开当前 BASE 是否值得，是否可能造成负迁移”。

## 证据充分性与 FRESH 闭环

1. “证据充分”定义为：原问题的每个必要 information need 都能指向当前累计证据，且没有未解决冲突；不能用“LLM 看起来能回答”替代。
2. 下一版建议使用三态：
   - `SUPPORTED`：所有必要项有证据，允许回答；
   - `REPAIRABLE_GAP`：能显式指出缺的实体/属性/关系/桥接证据，进入修复；
   - `CONFLICTED_OR_UNCERTAIN`：证据冲突、问题含糊或 judge 不稳定，继续核验或弃答。
3. 每轮保存 `needs -> evidence_refs` 支持图、结构化 gaps、contradictions 和 progress；先映射到现有 `sufficient / insufficient / unknown`，确认后再升级 schema。
   “累计证据”固定拆为：不可变 `EvidenceLedger`、去重后的 `ActiveEvidenceView`、`NeedSupportGraph`、逐轮 `ProgressDelta` 和 REUSE 前 `BranchCheckpoint`。它不是持续改写的摘要，也不是把所有 top-k 文本直接拼接。
4. FRESH 是同一 QueryEpisode 内的现场修复：始终以原问题 `q0` 为目标，根据累计证据的当前 gap 生成下一条 query；不调用跨题 ExperienceCard。
5. 默认终止建议：充分即答；总共最多 4 次检索（1 BASE + 最多 3 repair）；连续 2 轮无 gap closure/新证据、query/gap/证据循环、持续冲突或预算耗尽则 `STOP_ABSTAIN`。具体阈值只在 dev 校准。
   部分 need 已支持时只修复 `partial / missing / conflicted` 的残余 gap；候选 query 与既有 query 重复且没有新增实体/约束/关系/action intent 时执行前拒绝。最大轮数在 dev 比较 1/2/3/4 repair，选取接近最佳质量时成本最低的全局值，test 前冻结。
6. 回答前 coverage gate 是主路由；回答后的 claim support check 暂作可开关安全项，不把两个 judge 混成一个含糊分数。

参考边界：S2G-RAG 已覆盖累计句子证据、二值 sufficiency 与结构化 gap；ReflectiveRAG 已覆盖本题内 `Sufficient/Refine` 和边际改进停止；AIR 已覆盖围绕未支持 query terms 的迭代改写及无新词停止；2026-08 的 HALT 已覆盖预期 hop claims 的逐项证据匹配与覆盖停止；Self-RAG/SURE-RAG 可参考回答后的支持检查。因此“证据不足后再搜”“need coverage”“结构化 gap”都不是中心创新。

## 记忆设计

1. `QueryEpisode`：一次问题从原查询到最终回答/弃答的完整具体轨迹；保存实际 query、gap/support map、证据指针、成本和结果，可触及多篇文档。
2. `DocumentSession`：**严格的一文档一版本局部记忆**；保存该文档的 alias/术语、section/chunk landmarks、ambiguity/failure hotspots、`gap × action` 局部 benefit/harm 统计及来源 episode-turn 指针，文档结束后冻结。它不保存答案、完整 episode、文档正文或跨文档规则。
3. `ExperienceCard`：从 verified beneficial episodes 晋升的跨问题、跨文档程序记忆；保存去实体化的 query/gap pattern、前置/禁用条件、repair form/intent、evidence contract、可靠性账本和版本来源。
4. 所有完成的 episode 都冻结；错误/未知 episode 进入隔离审计档案。正确性决定能否晋升，不决定是否保存。
5. v1 在线只调用成功且经独立验证有增益的卡；失败 episode 不进入 serving memory。
6. 采用具体—局部—抽象表示：具体 `q -> q'` 永久留在 QueryEpisode；单文档导航知识放在精确版本的 DocumentSession；长期卡保存去实体化 operator/条件/证据契约，并指回来源 episode/turn。
7. 不保存完整 CoT、历史答案或文档正文到 active card；保存结构化 gap、动作、证据指针、成本和可验证结果。
8. 卡片合并或修订生成新版本，不覆盖来源或旧卡。
9. `serving memory` 统一称“在线可用库”：只有 active cards 和精确版本匹配的文档层可影响当前 query；`cold archive` 统一称“审计档案库”：保存冻结 episode、失败、隔离/退役卡和旧版本，默认不进入在线 prompt。
10. `forget/retire` 默认只从在线可用库移除，不物理删除原始记录；合并生成带 parent 的新版本。

三层的一行定义：QueryEpisode 记录“这道题发生了什么”；DocumentSession 记录“在这一篇文档里怎样找”；ExperienceCard 记录“跨文档仍可能怎样修”。三层本身不是创新，Useful Memories 与 SegMem-RAG 已覆盖相邻思想，RRM 明确提出不同职责的三层。09-06 最小实施暂不激活文档层，以上继续作为保留设计而非首版必须实现的清单。

## 成功与晋升

- `working_success`：修复后答对，但没有 DIRECT 对照；
- `beneficial_success`：同环境/预算下优于 BASE/DIRECT，或补回其缺失的 gold supporting facts；
- `trusted_success`：在独立 query/document 上重复有益且 conditional harm 低。

单次成功只能成为 exemplar/candidate；第 2 层才可生成 candidate card，第 3 层才可 active。

验证来源优先级：gold / 人工 > 多 judge 一致 > proxy。只有 Recall、相似度、QPP 或 sufficiency proxy 不能直接证明 trusted。

## REUSE 选择：可靠性与适用性分离

- `reliability` 回答“这张卡过去在独立题/文档上是否相对 BASE 反复有益、伤害率是否低”，是卡片级、慢更新历史统计；
- `applicability` 回答“可靠的卡是否适合眼前的 q0、gap、文档版本与 retriever 能力”，是每题重算的瞬时匹配。

08-24 候选四级流程为：reliability 硬门 -> applicability 约束 -> query/gap/action 与 QPP 排序 -> 执行后 evidence contract。09-05 复核后不再把这个顺序视为已确定最优方案：卡级小样本不能可靠估计低风险，首先对比历史特征、当前状态特征、条件门和联合选择；明确禁用条件可硬拒绝，其他分数的权重与门限在独立 calibration 上确定。未知不等于已证明不可靠。

每张卡保留**经验证的 paired 使用事件**账本，EMA 只表示近期趋势和漂移警报。首轮主实验冻结经验与选择策略，动态更新另用无未来泄漏的在线协议。Gold 缺失时 contract/支持判断只能作为标明来源的 proxy，不能写成真实 benefit/harm。REUSE 失约时保留审计与可回退状态；冲突新证据可能更准确，不能仅因来源是 REUSE 就删除。回滚仍计入所有调用成本。

RRM 已保存 applicability conditions、required evidence、query-adjustment patterns，并做衰减、合并和裁剪；这些字段与 top-k 淘汰不能归我们。GrowRAG 的候选增量必须落实为相对 FRESH / 现有经验选择的预测与决策收益，不能仅靠双轴术语、paired harm 或 contract 字段声称创新。

## 查询变换与 QPP

首版不发明新 query rewriter，把已有方法作为可选 operator/基线：

- LLM rewrite / Rewrite-Retrieve-Read；
- Query2doc expansion；
- HyDE（主要用于 dense retrieval）；
- decomposition；
- disambiguation、add constraint、bridge entity、evidence focus。

接口不采用一个混杂的扁平枚举，而分成：

- `QueryDecision = KEEP | TRANSFORM`；
- `transform_form = PARAPHRASE | EXPAND | DECOMPOSE | HYDE`；
- `repair_intent = DISAMBIGUATE | ADD_CONSTRAINT | BRIDGE_ENTITY | FILL_ATTRIBUTE | FILL_RELATION | EVIDENCE_FOCUS`。

LLM rewrite 描述生成手段；expansion/paraphrase/decomposition 描述变换形式；HyDE 更准确地说是检索表示变换。QPP 只作为当前候选的低成本预期检索质量特征，不判断 sufficiency、reliability 或最终答案正确性。v1 优先线上 pre-retrieval QPP，post-QPP 用作离线对照/分析。

## Prompt 与双控制器

v1 冻结三个 prompt：`state_judge`、`fresh_repair`、`experience_apply`。冻结包括模板、few-shot、模型版本、解码参数、JSON schema、阈值与 prompt version；开发集确定后测试期间不变。`state_judge` 只基于原问题、atomic needs 和当前 evidence IDs 判覆盖/gap，不生成 query，也不凭模型常识补答案。

Self-RAG 的 `IsREL/IsSUP` 主要启发文档相关性与回答 claim support；不能表述为“query 被文档支持”。GrowRAG 分开检查：执行前卡片是否可用、执行后 query 是否履行 gap/evidence contract、回答后 claims 是否被证据支持。

在线 `RetrievalController` 观察 EvidenceState、预算、QPP 和候选卡的 reliability/applicability，选择 `ANSWER / REUSE_TRANSFORM / FRESH_TRANSFORM / STOP_ABSTAIN`。它不只是两个失败分支。v1 用冻结规则/prompt。

异步 `MemoryLifecycleController` 在回答后选择 `NOOP / CREATE_CANDIDATE / PROMOTE / MERGE_AS_NEW_VERSION / QUARANTINE / RETIRE_FROM_SERVING`。QueryEpisode 每题结束自动冻结，不由 controller 决定是否保存。该分离参考但不同于 MemCon 把在线调用和 Consolidate/Forget/NoOp 放入同一 MDP。

以后训练的小 controller 只选择动作，不负责生成改写。训练标签来自 train/dev 上同题实际执行 BASE/REUSE/FRESH 后的 full-information oracle：选择达到质量阈值、成本最低且不产生 correct-to-wrong harm 的动作。先从规则、逻辑回归或树模型开始，不从 RL/bandit 开始。

## 环境处理

- 每个 episode 只保存 `run_context_id`，实际语料/切块/索引/retriever/reranker/LLM/prompt/judge 版本存在单独不可变注册表。
- v1 固定一个 BASE 环境，严格同环境复用。
- 长期卡只声明必要能力，例如 dense search 或 graph neighbor expansion；跨 retriever/corpus 是后续泛化实验，不要求首版同时解决所有变量。
- EMA/Kalman-inspired gain 只作为后续 priority/recent reliability 更新，不证明当前 applicability，也不是主创新。

## 当前待验证的中心问题与候选贡献

中心问题：**何时值得复用过去的查询修复？** 不再将“配对伤害指标”或“安全架构”写成已成立贡献。

1. 先证明在同一 q + evidence + gap、相同预算下，历史经验相对无记忆 FRESH 确有额外收益空间；
2. 再证明当前状态与历史迁移结果可以预测这个空间，而不只是检索到语义相似的卡；
3. 在相近修复收益、经验调用率或总成本下，比最近邻、ReFormeR/RRM 风格选择减少退化；
4. 对照具体 query pair 与抽象卡，不能预设抽象一定好；
5. 独立题、去近重复与跨数据集评价成立后，再接入完整多轮和动态记忆。

以上全部是可证伪研究假设，当前没有真实 QA 实验确认。三层、QPP、动作词表、衰减/top-k/merge/forget、sufficiency loop 是支撑模块。不能保证不存在相似工作，不使用“首次”表述。

## 已确认不能作为创新的内容

- Adaptive-RAG 式简单/复杂路由；
- 小模型、bandit 或 RL 控制检索；
- plug-and-play / frozen base / Agentic RAG；
- 显式 gap 后生成 query；
- query rewriting、HyDE、Query2doc、decomposition；
- episodic + consolidated 双记忆；
- applicability conditions、query-only reuse、成功/失败库、生命周期衰减；
- 经验驱动地选择 retriever/strategy。

## 关键近邻

- Adaptive-RAG：复杂度路由；
- C-3PO / SPARKLE：冻结底座上的 plug-and-play agentic controller；
- S2G-RAG / Skill-RAG：结构化失败/gap 到定向修复；
- RRM：跨任务程序检索经验、applicability、成功/失败库与 query-only reuse；
- ReFormeR：历史 query-reformulation pattern 的抽取与选择；
- MemCon：旁路 memory controller、NoOp 与在线记忆操作；
- ReMe：正负经验提炼、适应性复用与 utility 生命周期；
- SegMem-RAG：episodic/procedural/semantic memory 与经验路由；
- Useful Memories：持续 consolidation 会损坏记忆；
- GAM-RAG：sentence-level gain-adaptive retrieval memory；
- Experience-RAG Skill：场景 + 经验 + 检索策略路由的可插拔层；
- ERSkill：记忆检索技能、query router、experience trie 与 capability/deploy 双前沿；
- QPP Query Variant Selection：检索前/后选择 query variant。
- SIM-RAG：候选答案与累计证据的 Accept/Reject critic；不等同于回答前 coverage；
- How Memory Management Impacts LLM Agents：经验跟随可能传播错误，支持用未来独立任务验证记忆质量。
- HALT：expected hop claims 与累计证据覆盖停止；need coverage 已被直接覆盖，不能宣称为新贡献。
- DMQR-RAG / SAGE：多种 query rewrite strategy 与自适应/学习选择，动作词表和策略路由不是中心创新。
- When Continual Learning Moves to Memory：已成功题保留率与失败题新学会率、harmful reuse；correct→wrong 不是我们的新指标。
- LivingRAG：Graph RAG 上的经验增强、grounding/novelty 写入；支持度不等于未来效用。
- Useful Memories 08-29 v2：追加式与不可见旧摘要等新消融仍保留具体轨迹对照优势；仅版本化/禁止覆盖不保证消除抽象损坏。
- Think Then Rewrite / ReFeed：先显式分析或利用失败反馈再改写，以及只保留成功改写轨迹，均已有直接近邻。
- MaFeRw / AdaQR / RetPO / SELF-multi-RAG：多方面反馈、偏好训练与联合检索/改写控制，是以后训练 rewriter/controller 的主要对照。

完整矩阵见 `knowledge/literature/2026-08-22_方案B自适应路由与新颖性边界.md`；Query Transformation 最新查重见 `knowledge/literature/2026-08-24_QueryTransformation与REUSE四级门控_精读导图.md`。

## 数据与实验原则

沿用已确定的 gold 路线，并按方案 B 补充 turn-level 轨迹：

1. 小样本 TREC-RAG/QPP 资源：校验 query variants、retrieval metric 与 answer metric；
2. HotpotQA fullwiki + 2WikiMultiHopQA：主跨题、跨文档 repair transfer 与 harmful reuse；
3. MuSiQue-Ans：困难 gap 与多跳泛化；MuSiQue-Full 只在遵守其缺证据候选上下文设定时作拒答压力，不直接当开放全库不可回答 gold；
4. BEIR/BRIGHT/TREC DL：query operator 和跨检索器外部有效性；
5. RAGRouter-Bench：后期跨 Naive/Graph/Hybrid/Iterative BASE 验证，不是首个实现目标。

测试集不建库、不晋升、不选阈值、不调 prompt。所有动作必须在相同检索次数/top-k/context/token 预算下公平配对。

系统需要带 gold 的 source/calibration 数据来产生 FRESH 轨迹并建立经验，但 v1 不要求梯度训练：空记忆时先走 BASE/FRESH；source split 产生 candidate cards；独立 calibration query/document 校准 reliability/applicability/Judge/停止；held-out test 冻结 memory。在线自进化只能作为另一个按时间顺序、无未来泄漏的 prequential 协议。以后训练小 controller 时，标签来自 train/dev 上实际执行 BASE/REUSE/FRESH 的 full-information oracle。

## 最近工作顺序（09-05 建议，待讨论）

### 本日后续用户决定：先精读，再选中心切口

- 用户要求将最值得逐句精读及最接近的论文整合为 Zotero RDF，先阅读几天；不能假定用户已读完之前的清单，也不能在阅读期间继续扩建系统。
- 当前阅读入口：`zotero/curated/GrowRAG_逐句精读与直接近邻_精选20篇_2026-09-05.rdf`。20 篇是旧推荐的精选整合，含 12 核心、5 直接近邻、3 配套，20 条独立导读子笔记及一条总览；不含 PDF、不直接写 Zotero 数据库。
- 三个待选问题见 `knowledge/decisions/2026-09-05_阅读后待选择的三个研究切口.md`：①历史是否比同预算 FRESH 更值得选；②保留适用前提的最小经验表示；③少量 gold 校准下的谨慎准入。它们不是旧 A/B/C，也未被用户敲定为贡献。先做第 1 项的建议仍待讨论，不代表授权实验。
- 用户已明确选择“三天后自动继续一次”。本对话一次性跟进已建立：2026-09-08 22:50，Asia/Shanghai；automation id 为 `growrag`。仅复核文献、结合新增读书笔记、更新候选路线与项目记忆；不改运行代码、不装依赖、不运行付费实验、不推送 Git，不替用户决定。不得重复建立自动化或解释为持续后台研究。
- 发表核验、研究路线和数据协议仍以本日复核文件为准；精选导读优先于旧包未经修正的 Extra。引用数没有实时计量核验，不编造高引标签。

### 阅读讨论结束后的候选实施顺序

1. 读本轮近邻复核，确认中心问题与最小实验；不先继续扩展完整 Schema；
2. 固定一个文本 BASE、一个 FRESH、同一个 Judge 和预算，实际采集配对轨迹；
3. 从独立 source 建小型冻结经验库，保留具体 query pair 与抽象卡两种表示；
4. 在训练/开发子集跑 FRESH 与多个 REUSE 候选，估计离线 oracle 空间；
5. 对比便宜选择器及相似度/适用性/QPP 基线，独立校准；
6. 留出评估 benefit/harm/coverage/cost，失败则改表示或缩小任务，不继续堆模块；
7. 信号成立才接多轮、执行后核验、严格文档层和生命周期；
8. 再考虑小 controller 与第二种 BASE。

可用 HotpotQA 官方 train 内 4,000 题的预算切片：1,600 建库、1,400 影子迁移/未来训练、1,000 调参校准；官方 dev 另留 500 题作一次性 pilot 评价。40/35/25 是工程初值，不是论文给出的比例。更完整协议见同日数据集总表。

## 代码与仓库状态

- 现有 `src/growrag` 是路线 A 的可信复用核心，可作为方案 B 的 REUSE 子模块，不再代表完整系统。
- QueryEpisode、结构化 state、配对 VerificationEvent、ExperienceCard 和可信 Registry 的 v0 数据合同已实现；当前 DocumentSession 仍只是 v0 归组容器，真实 judge/QPP/controller/retriever 尚未接入。
- 2026-08-24 的三层记忆与 EvidenceState 是待确认 Schema v1；本轮只更新文档，没有提前改代码。
- 本地 `origin` 已指向 `https://github.com/Keep-Passionate/growRag.git`；2026-08-22 已完成首次发布，此后继续同步 `main`。
- GitHub 连接器不是 Git 提交/推送的必要条件；当前本机 Git 凭据已能完成仓库同步。
- 2026-08-24 本轮研究文档与 RDF 已提交到本地 `main`；向 GitHub 推送时 `github.com:443` 连接被重置/超时，因此新提交尚未进入远端。GitHub 插件安装请求已发出，但当前任务中尚未出现可调用的 GitHub 连接器；重新载入/完成界面授权后再检查。

## 更新规则

- 方向改变先更新 `knowledge/decisions/`，再同步本文件；
- 任何新论文若覆盖 provenance、paired harm 或 applicability，立即重做 novelty matrix；
- 所有 episode 完成即冻结，验证只追加；
- 每次实验先写假设、预算、公平对照与停止条件；
- 不使用“首次”措辞，直到投稿前完成系统检索与引用追踪。

## 当前详细讨论入口

- `knowledge/decisions/2026-09-05_重新评估_项目路线与贡献边界.md`
- `knowledge/literature/2026-09-05_可插拔经验系统与安全复用新颖性复核.md`
- `knowledge/literature/2026-09-05_四批阅读包发表状态复核.md`
- `knowledge/literature/2026-09-05_必读路线与RAG术语学习单.md`
- `knowledge/datasets/2026-09-05_历史论文数据集总表与划分依据.md`
- `knowledge/method/2026-08-24_证据充分性_三层记忆_QPP与双控制器_讨论稿.md`
- `knowledge/method/2026-08-24_累计证据_FRESH停止_REUSE动态可靠性与Judge_讨论稿.md`
- `knowledge/literature/2026-08-24_QueryTransformation与REUSE四级门控_精读导图.md`
- `knowledge/literature/2026-08-24_现有Zotero阅读包_发表层级审计.md`
- `knowledge/datasets/2026-08-19_Gold数据集与阶段规划.md`
- `zotero/current/GrowRAG_证据闭环_REUSE与QueryTransformation_2026-08-24.rdf`
