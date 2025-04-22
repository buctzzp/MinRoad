最短路径搜索可视化工具

基于 PyQt5 与 NetworkX 的交互式图搜索可视化应用，支持多种常见最短路径算法，通过动画展示算法过程，帮助理解搜索原理。

⸻

功能概览
	•	交互式图编辑：
	•	点击或拖动添加/移动节点
	•	选择节点对创建带权有向边（权重可自定义或自动计算欧氏距离）
	•	起点/终点设置：
	•	在界面上点击指定起点和终点节点
	•	算法选择与可视化：
	•	支持算法：DFS、BFS、A*、Dijkstra、SPFA
	•	动画演示搜索过程，包括节点扩展、入队、更新及最终路径高亮
	•	算法执行统计：路径长度、扩展节点数、运行时间
	•	随机生成示例图：
	•	网格布局的 30 个节点示例
	•	可选择包含负权边或只含正权边
	•	图结构持久化：
	•	保存／加载 JSON 格式的图（节点、边、位置、权重）
	•	背景图支持：
	•	在 QGraphicsScene 中加载并显示背景图片

⸻

环境与依赖
	•	Python 3.7+
	•	PyQt5
	•	NetworkX

可使用 pip 或 conda 安装：

# 使用 pip
pip install pyqt5 networkx

# 或者使用 Conda
conda install pyqt pyqtnetworkx networkx



⸻

快速开始
1.	克隆项目到本地：
git clone <仓库地址>
cd <项目目录>
2. 安装依赖（参考上文）  
3. 运行主程序：
   ```bash
python main.py
4. 在主窗口：
	•	通过工具栏或菜单切换到“添加节点”/“添加边”模式进行图编辑；
	•	设置起点和终点后，选择或切换算法，点击“开始搜索”观察动画；
	•	在“文件”菜单中保存或加载图结构；
	•	可通过“操作”菜单生成随机图或加载背景图片。

⸻

代码结构说明
	•	GraphView (QGraphicsView 子类)：
	•	维护 networkx.DiGraph 与图形元素映射
	•	实现节点/边绘制与点击响应
	•	提供搜索动画控制与步骤执行（定时器驱动）
	•	MainWindow (QMainWindow 子类)：
	•	菜单栏与工具栏构建
	•	算法选择与调度
	•	文件操作（保存／加载／加载背景图）
	•	算法实现：
	•	每个算法函数均返回 (path, steps)，steps 用于动画逐步展现
	•	支持负权图时禁用不适用算法并使用 SPFA

⸻

自定义与扩展
	•	权重计算：当前默认欧氏距离，可在 addEdge 中传入 custom_weight
	•	节点样式：节点半径、颜色、文本可在 GraphView.addNode 中修改
	•	动画节奏：在 animateSearch 中调整 step_interval 或最大总时长
	•	更多模块：可在主菜单中增加算法或工具栏按钮

⸻

许可协议

本项目采用 MIT License，欢迎 Fork、Star、贡献！