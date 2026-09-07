# GrowRAG 简明逻辑图

- figure_goal: 让用户理解原RAG外面的经验候选选择层，以及本轮实现边界。
- paper_claim: 只解释研究原型的组织，不表达性能或新颖性结论；不是最终论文方法图。
- figure_type: system-architecture
- mode: image（使用本地Graphviz精确生成，不调用图片API）
- panels: 新题在线候选选择；独立RAG；离线来源与冻结候选库。
- must_keep_labels: 原始问题、候选匹配、BASE、不改内部、下一批、本轮原型。
- data: not_applicable
- style_constraints: 白底、简短中文、蓝色标本轮候选匹配、灰色标原RAG；箭头从上往下；不把CANDIDATE自动画成可信REUSE。
- output_formats: dot, svg, png
- verification_checklist: 中文可读；BASE出口明确；返回原问题回答；原型匹配与自动真实路由分离；新经验仅下一批使用；无QPP或多轮已实现的误导。

文件 `render.cjs` 沿用项目既有本地Graphviz/Sharp渲染方式；参数为现有Node包目录。PNG供对话预览，DOT/SVG供编辑。
