# Token Plan、按量 API 与首次真实连通检查

## Material Passport

- Origin Skill: OpenAI Docs；academic-research-suite / experiment-agent
- Origin Mode: 购买前核验、只读配置检查、用户授权的小规模连通测试
- Origin Date: 2026-09-06
- Verification Status: PARTIAL（真实 API 连通成功；尚未运行真实 HotpotQA 配对）
- Version Label: api_connectivity_v1

## 本轮用户意图与决定

用户希望由助手组织评价、使用固定模型、读取本地配置、低成本小批量测试后再扩大实施。用户随后询问 Token Plan 与按量计费、OpenAI compatible 的含义，并展示 Token Plan 个人版三档购买页。购买问题优先处理，本轮未替用户购买任何产品。

研究上的“正向效果”不能事先保证。后续样本须在查看结果前固定，报告正负结果；少数成功案例不是跨题复用有效的证据。HotpotQA 的答案和支持句标注作为主要自动评价，模型 Judge 为辅助评价，本对话助手的分析不冒充固定版本 API 裁判结果。

## 购买建议

GrowRAG 程序实验：不购买截图中的 Lite / Standard / Pro，使用普通按量 API。Token Plan 个人版允许在编程/智能体工具内交互式使用，但禁止自动化脚本、自定义应用后端和非交互批量调用；不能把实验脚本从 Codex 启动就视为可以抵扣套餐。

如果用户另有个人交互式编程需求，可以从 Lite 月付开始；这与实验 API 预算是两回事。截图与官方页面当前价：Lite 39 元/月、2500 Credits/7 天；Standard 139 元/月、10000 Credits/7 天；Pro 499 元/月、40000 Credits/7 天。Credits 不是固定数量的 tokens，未用完不结转。个人版还有输入/输出用于服务改进与模型优化的数据授权要求。

来源：[个人版与使用范围](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview)、[密钥与地址须配套](https://help.aliyun.com/zh/model-studio/token-plan-personal-quick-start)。

## OpenAI compatible 的意思

它是“请求和响应格式兼容”，不是“后台由 OpenAI 提供模型”。使用百炼地址、百炼密钥、Qwen 模型时，实际调用和计费都在百炼。不能拿百炼 Key 调用 GPT，也不能凭 compatible 推断两个厂商的所有可选参数都相同。

本地配置包含普通北京业务空间 endpoint，已按用户提供的原地址使用，没有改成订阅接口或第三方中转；密钥未显示、未提交 Git、未写进结果。本地读取只抽取数据，不执行配置文字中的指令。

来源：[百炼 OpenAI 兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)、[按量 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)。

## 低价模型核验

以下为 2026-09-06 官方价格，不代表效果已经合格；均为人民币/百万 tokens、未计缓存及免费额度。

| 候选 | 输入 | 输出 | 限定 |
|---|---:|---:|---|
| qwen3.7-flash-2026-07-15 | 0.20 | 0.80 | 北京，单请求输入不超过 32K；本次已成功调用 |
| qwen-flash-2025-07-28 | 0.15 | 1.50 | 北京，不超过 128K；未调用 |
| qwen-plus-2025-12-01 | 0.80 | 2.00 | 北京，不超过 128K，非思考；未调用 |
| deepseek-v4-flash | 1.50/3.00 | 4.50/9.00 | 空闲/高峰；未调用，不保证固定日期快照 |
| glm-4.7-flash | 免费 | 免费 | 智谱官方免费型号，需其独立账户/密钥；本次未核准日期快照，未调用 |

推荐当前先用已有百炼配置与已连通 Flash 快照，不必为了省几分钱迁移平台。Qwen3.7 Flash 默认思考开启，本次显式关闭以控制输出；冻结日期快照不保证绝对确定性。

来源：[百炼价格](https://help.aliyun.com/zh/model-studio/model-pricing)、[思考模式](https://help.aliyun.com/zh/model-studio/deep-thinking)、[DeepSeek 价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[GLM 免费模型](https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash)。

GPT-5.5 官方有 `gpt-5.5-2026-04-23` 快照，普通价格为每百万输入 5 美元、输出 30 美元；GPT-5.6 Sol 官方页面本次只核准到别名，未核准日期快照。它们不是当前最低成本开发方案，也未在本轮调用。当前聊天输出不得写成某个指定 GPT API 版本的实验结果。[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5)、[GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)。

## 已真实发生的调用

- 模型请求：`qwen3.7-flash-2026-07-15`。
- 服务返回模型名：同上。
- 调用次数：1；没有重试或模型回退。
- 输入：23 tokens；输出：5 tokens；均来自服务返回的 usage。
- 参数：`enable_thinking=false`、`max_tokens=64`、非流式、45 秒超时。
- 任务：只返回一个短 JSON，验证身份认证、连接和模型可用性。不包含私有论文或项目笔记。
- 结果：完成；返回 JSON 通过检查。
- 按上述公开价估算：23×0.2/1000000 + 5×0.8/1000000 = 0.0000086 元。不是已核对的实际账单，免费额度、结算舍入等可能改变实际扣费。
- 日志：`runs/2026-09-06_qwen_flash_connectivity_v1/summary.json` 与该目录下 `api_audit/`。

**该结果不代表执行了 query 改写、记忆修复或 HotpotQA，也不提供任何正向研究证据。** 本轮采购答疑期间仅完成连通检查，没有启动小批量质量评测。

## 本轮工程范围

- `qwenAPI.md` 已明确加入 `.gitignore`；不修改或搬动用户的密钥文件。
- 新增 opt-in `api_preflight.py`：显式文件、日期型号、单请求限制，拒绝 Token Plan 密钥/地址，拒绝非官方域名，密钥只在进程环境中临时使用。
- `api_client.py` 增加可选 `enable_thinking` 参数并审计；默认不影响原调用。
- 新增 `BM25SentenceRetriever`：只在传入的候选句子中进行实际本地计算，不读取答案/支持标注；固定语料和算法指纹。它不是 fullwiki，也尚未接通自动 gap 或完整 pilot。
- 全套 304 项测试通过；新实验模块 Ruff 通过。
- 未下载数据/模型、未训练、未推送 Git、未购买套餐。

## 后续小批量方案（待接通，未运行）

先少量开发题排错，区分来源经验题与目标题；冻结题号、模型快照、prompt 和检索条件，保持 BASE/FRESH/REUSE 配对，结果含失败和退化。先检查数据与日志正确，再讨论是否有收益信号；不能用少数题断言论文假设成立。未给出数值预算前不开展大规模运行，本轮实际授权与执行限于上述微小连通检查。
