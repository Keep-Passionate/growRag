# 上游代码

## 2026-09-27：ReFormeR参考文件已取得

- 来源：`https://github.com/aminbigdeli/ReFormeR/tree/72e52450a922bc0051b5b39c3a6307186a555137`。
- 本地：`reformer_author_snapshot/aminbigdeli-ReFormeR-72e5245/`，仅下载core、prompt manager、prompts、patterns与README；不安装上游依赖、不运行main。
- `src/growrag/experiments/reformer_api.py`保存四个运行文件的SHA，只有匹配后才加载已审阅方法；API替换vLLM，非作者训练selector复现。
- README有MIT徽章，但缺完整LICENSE、GitHub API licence=null；继续保留在ignored external，不向公开仓库转存源码或示例库。
- 下载需固定commit，不能换成main后忽略哈希失败。库中的示例不默默修正；方法改进要另立版本与对照。

本目录用于浅克隆 QPP-4-RAG、ReFormeR、QueryGym 等公开上游。目录内容不提交 Git；上游版本应在实验清单中记录 commit hash。

官方地址：

- https://github.com/Narabzad/QPP-4-RAG
- https://github.com/aminbigdeli/ReFormeR
- https://github.com/ls3-lab/QueryGym

历史状态（2026-08-19）：地址已核对，但当日本机到 GitHub 443 端口连接超时，未克隆。重试命令见根目录 `GITHUB_SETUP.md`。

## 2026-09-26：S2G 作者快照已取得

- 官方来源：https://github.com/nianaaa/S2G-RAG
- 固定提交：`5d842a67a0a99a7b545bbad0dc402ceaae0e5eff`
- Git 连接中断后，使用 GitHub 对应提交的 ZIP 归档；保留于 `s2g-5d842a6.zip`，解压目录 `s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6/`。
- 作者目录和归档均被 Git 忽略；只将我们编写的审计工具、测试与说明提交。当前未找到代码 LICENSE，不向公开仓库转存作者源码或修改版。
- 只检查24个Python文件语法，并执行4个锁定SHA的原始纯函数的6项检查。尚未运行完整QA、训练或加载Judge权重。
- 结果位于 `runs/2026-09-26_s2g_author_audit/preflight.json`；资源缺口、数据隔离与下一步见 `docs/reproduction/2026-09-26_S2G作者代码迁移与历史路由决策.md`。
- 可用 `python -m growrag.experiments.s2g_upstream_check --upstream external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6 --output <新的审计文件路径>` 重做只读源文件检查。输出文件若存在会拒绝覆盖，不下载模型、不访问API。
