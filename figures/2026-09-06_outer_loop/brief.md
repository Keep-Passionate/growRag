# 最小代码架构图

- figure_goal：说明外置查询层如何包围独立 RAG，代码责任如何分开。
- paper_claim：仅描述工程接口；状态机、QPP、统一动作均不冒称创新。
- figure_type：system-architecture；mode：image（以 Graphviz 精确绘制，不生成栅格草图）。
- panels：问题、外层控制、改写器、RAG、反馈与停止；经验和独立 QPP 用虚线。
- must_keep_labels：BASE/FRESH/REUSE、原始问题、独立 RAG、检索前特征、外部反馈。
- data：not_applicable，无实验指标。
- style_constraints：中文短标签、白底、蓝色执行链、灰色底座、虚线未接入模块。
- output_formats：DOT、SVG、PNG。
- verification_checklist：原问题不变；经验不直接进入 Reader；QPP 不冒称已控制在线策略；证据不可见时明确退化；反馈循环是完整 RAG 调用，成本高于只追加检索。
- 语义差异：本轮外层可以累计观察供下一次规划，但不能保证任意黑盒 RAG 的内部回答器读取跨轮累计证据。最小适配器每轮只用该轮检索结果回答；以后若底座支持 context 输入再明确启用累计阅读。
