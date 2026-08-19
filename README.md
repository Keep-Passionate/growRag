# GrowRAG

GrowRAG 目前处于研究问题收敛与可行性验证阶段。当前最值得验证的候选方向是：

> 在第一次正式检索前，只有当一条历史查询变换本身可靠、且对当前问题的适用性风险足够低时才复用；否则保持原查询不动。

这不是“新 RAG 范式”的既定结论。第一阶段先测量历史查询变换是否真的存在跨题迁移空间，以及能否控制 `DIRECT 原本正确、REUSE 反而错误` 的非对称伤害。

## 当前入口

- [项目当前记忆](knowledge/CURRENT_PROJECT_MEMORY.md)
- [方向候选与待选择事项](knowledge/decisions/2026-08-19_ERM继承路线与待选择.md)
- [ERM 继承与安全复用方案](knowledge/method/2026-08-19_ERM继承与安全复用.md)
- [经验记忆与 EMA 备忘](knowledge/method/2026-08-19_经验记忆与EMA.md)
- [Gold 数据集与阶段规划](knowledge/datasets/2026-08-19_Gold数据集与阶段规划.md)
- [实验阶段与止损门](knowledge/experiments/2026-08-19_实验阶段与止损门.md)
- [当前 Zotero 导入包](zotero/current/README.md)
- [旧工程归档说明](archive/legacy_2026-08-19/MANIFEST.md)

## 代码状态

`src/growrag` 是 2026-08-19 新建的最小研究脚手架。旧版“账本—验证门—经验注入”代码已经退出活动工程，保存在 `archive/legacy_2026-08-19`，不得再作为新方法的默认依据。

当前代码只定义：

- `DIRECT / REUSE / FRESH` 三种实验动作；
- 查询变换与环境版本的中立数据结构；
- 成对比较 `DIRECT` 与 `REUSE` 的 benefit / harm 标签；
- 可复现研究所需的最小测试。

它还没有实现最终选择器、ERM 索引更新、EMA、LLM 调用或数据下载。

## 本地启动

推荐使用 Python 3.11。当前机器已创建并安装好独立的 `.venv`：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

如需重新创建，可以使用本机已有的 Python 3.11：

```powershell
D:\develop\Anaconda3_2025.12-1\Anaconda_env\envs\srvtools\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

也可以在 Conda 网络和渠道配置正常时使用：

```powershell
conda env create -f environment.yml
conda activate growrag
pytest
```

2026-08-19 本机 Conda 全局配置含失效旧渠道，且官方渠道下载中断；本轮没有修改全局设置，改用上述 `.venv` 成功安装。

## 仓库边界

- `data/`、`runs/` 和 `external/` 的大文件不提交 Git。
- `archive/` 只跟踪说明文件，不提交旧代码、PDF 或历史运行产物。
- GitHub 远端尚未创建；原因和下一步见 [GITHUB_SETUP.md](GITHUB_SETUP.md)。
