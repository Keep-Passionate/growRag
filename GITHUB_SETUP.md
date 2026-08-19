# GitHub 设置说明

本地 Git 仓库可以直接初始化，但当前机器没有安装 GitHub CLI（`gh`），而且远端仓库的公开性尚未由项目负责人决定，因此本轮不会擅自创建或发布 GitHub 仓库。

## 建议

- 研究方向未冻结前，远端仓库先设为 **Private**。
- 仓库名可暂用 `GrowRAG`，待论文主张确定后再决定是否改名。
- `archive/`、`external/`、`data/` 和 `runs/` 已被忽略，不会把旧 PDF、数据集或上游代码误提交。
- 上游仓库只作为可替换的实验参考，不复制到 `src/growrag`。

## 创建远端后连接

在 GitHub 网页新建一个空仓库，然后执行：

```powershell
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git push -u origin main
```

如以后安装 GitHub CLI，也可以：

```powershell
gh auth login
gh repo create GrowRAG --private --source . --remote origin --push
```

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

2026-08-19 已核对这三个官方地址。本轮命令行到 `github.com:443` 的连接超时，因此目录尚未下载；网络恢复后重试即可。
