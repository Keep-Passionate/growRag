# GitHub 设置说明

2026-09-06 最新：活动分支为 `codex/query-card-actions-v1`，0.3.0 功能提交 `3b8da81` 已推送；标签 `v0.3.0-research.1` 可作为回退点。完整 497 项本地/干净副本测试通过，远端 Python 3.11/3.12 CI 成功。[验证记录](knowledge/experiments/2026-09-06_前置三分支_版本验证.md)。下文 Hotpot pilot 为此前历史，不是当前活动分支。

远端仓库已经由项目负责人创建，现已包含代码与研究笔记：

`https://github.com/Keep-Passionate/growRag.git`

本地 `origin` 已指向该地址。2026-08-22 完成首次推送；2026-09-06 通过现有 Git 凭据将新代码推送到 `codex/hotpot-pilot-v1`，本地该分支跟踪远端同名分支，没有自动合并 `main`。无需额外连接器即可完成这次版本控制，也没有安装插件或导出凭据。

实际运行代码提交为 `3d58c39`；随后 `09d3a22` 修复干净检出的手写 CLI 样例漏提交。第一次 CI 的失败是真实缺文件，不是模型实验失败；已用干净 Git 文件导出复现并精准放行该公共样例。数据、运行日志和密钥仍保持忽略。

`09d3a22` 的 [GitHub CI](https://github.com/Keep-Passionate/growRag/actions/runs/34012633506) 已核实成功，覆盖 Python 3.11/3.12。

仓库内已经提供 `.github/workflows/ci.yml`；连接并推送远端后会自动在 Python 3.11/3.12 上运行测试、代码检查和 CLI 冒烟测试。

## 建议

- 当前远端是 Public；提交前继续依靠 `.gitignore` 和审阅避免上传数据、PDF、密钥与运行产物。
- 仓库名可暂用 `GrowRAG`，待论文主张确定后再决定是否改名。
- `archive/`、`external/`、`data/` 和 `runs/` 已被忽略，不会把旧 PDF、数据集或上游代码误提交。
- 上游仓库只作为可替换的实验参考，不复制到 `src/growrag`。

## 当前连接

后续同步使用：

```powershell
git push -u origin codex/query-card-actions-v1
```

如以后安装 GitHub CLI，也可以：

```powershell
gh auth login
git push -u origin codex/query-card-actions-v1
```

不要 force push。后续推送前先正常 fetch/pull，避免覆盖用户从网页或其他机器提交的内容。

## 当前准备跟踪的上游

| 上游 | 用途 | 本项目关系 |
|---|---|---|
| `Narabzad/QPP-4-RAG` | query variant、QPP、oracle、TREC-RAG 复现骨架 | 主要评测参考 |
| `aminbigdeli/ReFormeR` | 历史成功 query pair 到 reformulation pattern | 主要方法基线 |
| `ls3-lab/QueryGym` | 查询改写方法与提示词的统一复现 | 工具参考 |

这些仓库放在 `external/`，不与本项目源码混合，也不会被提交到远端。

## 上游浅克隆

```powershell
git clone --depth 1 https://github.com/Narabzad/QPP-4-RAG.git external/qpp-4-rag
git clone --depth 1 https://github.com/aminbigdeli/ReFormeR.git external/reformer
git clone --depth 1 https://github.com/ls3-lab/QueryGym.git external/querygym
```

2026-08-19 已核对这三个官方地址。当前未下载，是因为 M0 数据合同不依赖这些上游；进入 M1 复现 query operator/基线时再按需浅克隆。
