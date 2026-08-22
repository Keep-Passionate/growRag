# GitHub 设置说明

远端仓库已经由项目负责人创建并核实为空：

`https://github.com/Keep-Passionate/growRag.git`

本地 `origin` 已指向该地址。当前机器的命令行到 `github.com:443` 连接超时；GitHub 连接器安装建议已经发出，但仍等待用户在界面中确认授权，所以尚未完成首次推送。

仓库内已经提供 `.github/workflows/ci.yml`；连接并推送远端后会自动在 Python 3.11/3.12 上运行测试、代码检查和 CLI 冒烟测试。

## 建议

- 当前远端是 Public；提交前继续依靠 `.gitignore` 和审阅避免上传数据、PDF、密钥与运行产物。
- 仓库名可暂用 `GrowRAG`，待论文主张确定后再决定是否改名。
- `archive/`、`external/`、`data/` 和 `runs/` 已被忽略，不会把旧 PDF、数据集或上游代码误提交。
- 上游仓库只作为可替换的实验参考，不复制到 `src/growrag`。

## 当前连接

本地 `origin` 已配置；网络/授权恢复后只需执行：

```powershell
git push -u origin main
```

如以后安装 GitHub CLI，也可以：

```powershell
gh auth login
git push -u origin main
```

不要 force push。远端当前虽为空，首次推送前仍应再次 fetch/核对，以防用户期间从网页添加 README 或 License。

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

2026-08-19 已核对这三个官方地址。2026-08-22 再次确认命令行到 `github.com:443` 仍连接超时，因此目录尚未下载；网络或 GitHub 连接器授权恢复后重试即可。
