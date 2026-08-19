# 文件化可信复用示例

这个目录演示 GrowRAG v1 的完整边界，但不会真正访问检索器或生成答案。

- `config.toml`：运行环境、候选数、可信阈值和统一预算；
- `memory.json`：一条已经通过独立校准并处于 ACTIVE 状态的历史经验；
- `requests.jsonl`：三个当前问题；
- `plans.json`：预先为当前问题生成的候选查询。

三个预期结果：

1. `target-reuse`：历史可靠、当前适用、计划合法，因此 REUSE；
2. `target-not-applicable`：历史可靠但当前结构不匹配，因此 DIRECT；
3. `target-missing-plan`：本来适合复用，但没有当前题计划，因此安全回退 DIRECT。

运行命令见项目根目录 `README.md`。
