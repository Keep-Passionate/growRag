# GrowRAG 两张论文图：构图讨论 v1

日期：2026-09-06。由用户要求先图后代码；本轮不扩展业务代码、不训练、不新增 RAG API 实验。

## 图 1：系统架构

- figure_goal：解释跨题经验怎样进入查询修复，以及训练经验库和评价题之间的边界。
- paper_claim：设计目标是让历史提供搜索过程，而当前证据提供回答事实；不是已证明安全或有效的结果声明。
- figure_type / mode：system-architecture / image。
- panels：上部训练期经验构建；下部新题运行期；后置文档层仅脚注。
- anchor：查询修复层的 BASE / FRESH / REUSE 三路，以及独立的当前证据 Reader。
- panel roles：方法定义、信息流与边界说明，不放准确率或优势图。
- must_keep_labels：QueryEpisode、ExperienceCard、Frozen memory、BASE、FRESH、REUSE、Current evidence、Reader、PROPOSED。
- data：不适用，无数量或性能图。
- style_constraints：白底、英文短标签、浅蓝与青色；候选控制器/语义支持检查为琥珀虚线；横向排版。
- output_formats：PNG 构图稿、提示词、中文图注。不是已生成可编辑矢量源。
- supplement：模型参数、预算、完整停止规则、DocumentSession、EMA 不塞入主图。
- verification_checklist：训练 gold 不进入在线决策；历史不直通 Reader；不用经验仍可 FRESH；BASE 绕过追加检索；当前 prototype 的三支配对不冒充已部署路由；当前仅单次修复；在线评价不回写源库。
- 图注逻辑：实线模块已接通，琥珀虚线为候选改进。评测时执行多个分支以得到配对反馈；未来部署时才选择一支。所有输出回答原问题。
- Results 对齐：这里只能配合“方法总览/系统定义”，不能配合“验证了可信性”的结果主张。

## 图 2：具体方法

- figure_goal：解释候选机制“不要整张套用或整张扔掉，按当前证据决定哪些经验步骤可用”。
- paper_claim：这是待验证假设；既有条件门、状态化工作流、QCR 重绑定与带守卫补丁均非我们的原创。
- figure_type / mode：algorithm-workflow / image。
- dominant message：Partial reuse, grounded in current evidence；醒目标注 Proposed method — not implemented / novelty unconfirmed。
- panels：(a) 来源经验中的有依赖操作；(b) 当前问题、当前证据与逐步执行状态；(c) 同状态、同预算 FRESH 对照，分开看证据、答案与费用。
- anchor：一张小表把步骤映射为 already supported / ready / waiting；历史实体值不参与当前绑定。
- illustrative example：虚构作品 Silver River、作者 Mira Chen；当前证据 e1 已支持作品作者，但没有其大学及建校年份。步骤 1 查作者被跳过；步骤 2 查大学可用当前作者绑定执行；步骤 3 查建校年份因大学未确认暂不执行。不是 Hotpot 题或真实实验。
- allowed execution：只执行当前可执行的步骤；未知前提可用 FRESH 查证，但同样计算预算；冲突需核验或弃用，不凭旧值补全。
- stop：支持充分、有效证据无增益、查询重复、预算耗尽；证据不足不因停止而被标为正确。
- no invented results：不填任何虚构准确率、提升百分比、因果保证或已训练控制器。
- scope：QPP/EMA 后置；当前 pilot 每支至多一次修复，图中再次检查属于待确认的未来机制。
- output_formats：PNG 构图稿＋完整提示词＋中文逐项图解；通过图审后再制作可编辑的投稿级矢量稿。
- verification_checklist：例子输入值均来自当前 q/E；等待动作无查询输出；金标准只用于离线评价；FRESH/REUSE 从相同 E0 独立分支；Reader 不读取历史轨迹；不得把普通条件检查宣称首次。
