# 正式开发首批：真实 PRE 来源与冻结经验比较

版本 0.4.0；用户已同意首批累计 5 元估算预算。结果以实际运行文件为准，本文不预先宣称实验成功。

运行前验证：598 项本地测试通过，静态与格式检查通过。新增 101 项测试覆盖独立来源、清单、防泄漏、无卡与预算边界；测试替身不计作真实模型效果。

真实首批已于9月6日完成：[结果与边界](../knowledge/experiments/2026-09-06_PRE真实建库与比较_结果.md)。32来源1卡，16目标无卡，REUSE未执行；145次API估算0.0209324元。下面的真实命令是历史协议，不应自动重新收费运行。

## 新增的最小连接

```text
来源题（仅 Hotpot train 来源角色）
   ├─ 原查询 → RAG → BASE 记录
   └─ 原问题独立改写 → RAG → FRESH 记录
                ↓ 运行结束后离线评价
        PreSourceRecord：两支并列，不伪造串行检索
                ↓ 源题改善才提取通用操作
        ExperienceCard：统一外壳＋ParaphraseBody
                ↓ 来源做完后冻结
新开发题 → query-only 候选匹配 → 选择候选或无卡
   ├─ BASE 原查询（必需对照）
   ├─ FRESH 不看历史独立改写（必需对照）
   └─ REUSE 只读候选卡改写（无卡则明确未执行）
                ↓ 每支立即保存，之后才读 gold
           逐题报告＋完整批次报告＋唯一预算账本
```

源题原查询和改写的结果互不作为对方输入；目标阶段不更新源库。`PreSourceProvenance` 是新来源类型，未放宽旧 Episode/registry，也不能绕过旧规则把卡晋升 ACTIVE。

主要文件：

- `experience/pre_sources.py`：独立来源记录、离线录入、统一卡和只读视图。
- `experiments/pre_manifest.py`：验证现有数据/划分/哈希，固定最多32来源＋16目标。
- `experiments/pre_pilot.py`：建库、模型提卡、候选匹配与批量对照。
- `experiments/pre_report.py`：完成数量、胜平负、误用与费用汇总。
- `experiments/run_pre_pilot.py`：显式真实启动；无网络参数时只显示计划，不读密钥。

## 什么情况下来源可以成为候选卡

源题两支均完整，实际查询有变化，FRESH 答案匹配 gold；答案和支持召回都不退化，且至少有一项改善。答案改善与证据改善分开存储：不是每张卡都必然有新增支持句，更不是严格因果证明。

模型提卡只看源问题和实际新查询，不接收源答案、gold、支持句或结果分数。抽出问题模式、适用/禁用条件和语义改写正文；结构检查不是语义去事实的证明，仍需逐卡人工检查。来源观察不计入跨题 matched_trials，卡仍是 candidate。

当前用来源题内容词 Jaccard（两个词集合重合比例）挑候选，去掉固定常见词；零重合可以不选。自然语言条件还没有独立判断器，不称为“可信选择器”。展示的 LEXICAL_POLICY 是执行前选定 BASE 或 REUSE 的诊断路线，不多付费运行第四支。

## 数据和固定设置

- 继续现有 Hotpot train 前200条开发数据；原163/21/16角色不变。从固定角色顺序取32来源与16校准目标，不看 gold 分数选择题。
- 官方 dev/test 未使用。这16题有旧试跑或 QPP 开发曝光，明确是回归/开发，不是盲测；候选池 BM25 也不是 fullwiki。
- 固定 `qwen3.7-flash-2026-07-15`，关闭思考，`temperature=0`，每次最多768输出 tokens，每题各支 top-k=4。温度为0不保证远程推理逐字可复现。
- 按2026-09-06[官方模型页](https://help.aliyun.com/zh/model-studio/text-generation-model/)与[北京价格页](https://help.aliyun.com/zh/model-studio/model-pricing)核验，短档输入0.20元/百万、输出0.80元/百万。不预设缓存/赠送折扣，原业务空间权限由实际返回核查。
- 整批最多256请求；计划协议最多208请求。5元是源题、提卡、目标全部共用的累计估算上限。每次输入最多24000 UTF-8字节、单请求45秒超时、整批新增调用限1800秒。

只有同一个串行预算客户端，没有为每个独立 RAG 发放新额度。模型不符、用量未知、网络或预算异常阻止新增付费请求；不自动重试、不换模型、不购买Token Plan。估算不是厂商结算保证。

## 启动与留痕

先仅检查：

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments.run_pre_pilot --manifest data/hotpotqa/train_preview_200_v1/manifest.json --output runs/2026-09-06_pre_pilot_32_16_v1
```

只有预算已授权、代码已提交且工作区干净时，才在相同命令后增加：

```text
--api-config qwenAPI.md --allow-network
```

此入口仅支持普通北京百炼配置；密钥不打印、不写入计划或Git，专用环境变量结束后恢复。已存在的输出目录拒绝重跑。

输出包括启动计划、冻结清单、源题记录/录入理由、候选卡、冻结库哈希、逐题三支记录、真实API审计、批次结果、汇总JSON与易读MD。每支保存后才启动下一支；磁盘失败停止流程。若进程中断，保留残存记录并人工核对，首版**不提供自动恢复或重发**。

未执行的 REUSE 保持未知，不能拷贝 BASE 分数充当它的成绩。报告同时展示正确数、完成数和全部题数，不只挑成功题计算均分。

## 首批之后

先逐卡核查是否过度抽象、来源实体是否残留，再逐题比较 REUSE−FRESH、BASE正确被改错、选择覆盖和费用。若没有额外作用，记录负结果；不要换掉坏题或偷偷改规则。下一步才比较更好的适用性/表示与 QPP，小模型训练、多轮和文档层仍后置。

## 0.4.1：不收费的来源检索对比

`experiments/retrieval_probe.py` 对同一来源题运行本地 BM25：比较原查询、实际成功改写和人工拆出的查询成分。先核对原 BASE/FRESH 的有序证据与归档完全一致，再保存新诊断；不调用 Reader，不读取 API 配置，不给人工查询填入原答案成绩。

```powershell
.\.venv\Scripts\python.exe -m growrag.experiments.retrieval_probe --manifest data/hotpotqa/train_preview_200_v1/manifest.json --source-record runs/2026-09-06_pre_pilot_32_16_v1/sources/025/source_record.json --output runs/2026-09-07_pre_source_retrieval_probe_v1/probe.json --candidate entities_only "Kim Clijsters Mary Pierce" --candidate entities_age "Kim Clijsters Mary Pierce age" --candidate entities_vs "Kim Clijsters vs Mary Pierce" --candidate without_tennis "Between Kim Clijsters and Mary Pierce, who is older?"
```

上述诊断已运行，输出保留且不覆盖。四个人工候选为事后探索，不是预注册的方法比较；只证明这次本地排序如何变化，不证明新答案或跨题效果。0.4.1 本地 633 项测试、静态和格式检查通过；本次后续无新增 API。
