import sys, math, heapq, time
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QGraphicsEllipseItem, QGraphicsLineItem, QAction, QFileDialog,
                             QInputDialog, QMessageBox, QGraphicsTextItem)
from PyQt5.QtGui import QPen, QBrush, QColor, QPixmap
from PyQt5.QtCore import Qt, QTimer, QPointF
import networkx as nx

##############################################
# 自定义图显示区域，继承自 QGraphicsView
##############################################
class GraphView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphView, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 设置白色背景
        self.setBackgroundBrush(QBrush(Qt.white))
        
        # 用 networkx 保存图结构（改为有向图）
        self.graph = nx.DiGraph()
        # 用于保存节点的 QGraphicsEllipseItem
        self.node_items = {}           # node_id -> QGraphicsEllipseItem
        # 保存所有边（QGraphicsLineItem）
        self.edge_items = []
        # 节点坐标（用于 A* 中的启发式计算）
        self.node_positions = {}
        
        # 当前操作模式（可选：'add_node', 'add_edge', 'set_start', 'set_goal', 'edit_weight'）
        self.mode = None
        self.temp_edge_start = None  # "添加边"模式下，记录第一个被点击的节点
        
        # 记录起始、目标节点 id
        self.start_node = None
        self.goal_node = None
        
        # 搜索可视化：存储算法中记录的步骤，借助定时器逐步展现
        self.search_steps = []
        self.final_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.animateStep)
        self.current_step_index = 0
        
        # 自动编号，初始节点 id 为 0
        self.next_node_id = 0
        
        # 创建默认的示例图
        self.create_example_graph()

    def setMode(self, mode):
        """设置当前操作模式，并重置临时变量"""
        self.mode = mode
        self.temp_edge_start = None

    def clearSelectionColors(self):
        """重置所有节点颜色为默认（浅蓝色）"""
        for node_id, item in self.node_items.items():
            item.setBrush(QBrush(QColor("lightblue")))

    def addNode(self, pos):
        """在 pos（QPointF）处添加一个新节点，自动编号"""
        node_id = self.next_node_id
        self.next_node_id += 1
        radius = 15
        ellipse = QGraphicsEllipseItem(-radius, -radius, 2*radius, 2*radius)
        ellipse.setBrush(QBrush(QColor("lightblue")))
        ellipse.setPen(QPen(Qt.black))
        ellipse.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        ellipse.setPos(pos)
        self.scene.addItem(ellipse)
        
        # 添加节点编号文本
        text = self.scene.addText(str(node_id))
        text.setDefaultTextColor(Qt.black)
        text.setPos(pos.x() - text.boundingRect().width()/2,
                   pos.y() - text.boundingRect().height()/2)
        
        self.node_items[node_id] = ellipse
        self.node_positions[node_id] = (pos.x(), pos.y())
        self.graph.add_node(node_id)
        return node_id

    def addEdge(self, node1, node2, custom_weight=None):
        """在 node1 与 node2 之间添加一条有向边，边权以两节点之间的欧氏距离计算或使用自定义权重"""
        if self.graph.has_edge(node1, node2):
            return
        pos1 = self.node_items[node1].pos()
        pos2 = self.node_items[node2].pos()
        
        # 计算箭头的角度和位置
        dx = pos2.x() - pos1.x()
        dy = pos2.y() - pos1.y()
        angle = math.atan2(dy, dx)
        
        # 调整箭头终点，使其不覆盖节点
        arrow_length = 20
        radius = 15  # 节点半径
        end_x = pos2.x() - radius * math.cos(angle)
        end_y = pos2.y() - radius * math.sin(angle)
        
        # 创建箭头线
        line = QGraphicsLineItem(pos1.x(), pos1.y(), end_x, end_y)
        line.setPen(QPen(Qt.black, 2))
        self.scene.addItem(line)
        self.edge_items.append(line)
        
        # 添加箭头头部
        arrow_angle = math.pi / 6  # 30度
        arrow_p1_x = end_x - arrow_length * math.cos(angle + arrow_angle)
        arrow_p1_y = end_y - arrow_length * math.sin(angle + arrow_angle)
        arrow_p2_x = end_x - arrow_length * math.cos(angle - arrow_angle)
        arrow_p2_y = end_y - arrow_length * math.sin(angle - arrow_angle)
        
        arrow_head1 = QGraphicsLineItem(end_x, end_y, arrow_p1_x, arrow_p1_y)
        arrow_head2 = QGraphicsLineItem(end_x, end_y, arrow_p2_x, arrow_p2_y)
        arrow_head1.setPen(QPen(Qt.black, 2))
        arrow_head2.setPen(QPen(Qt.black, 2))
        self.scene.addItem(arrow_head1)
        self.scene.addItem(arrow_head2)
        self.edge_items.append(arrow_head1)
        self.edge_items.append(arrow_head2)
        
        # 如果提供了自定义权重，使用它；否则计算欧氏距离
        if custom_weight is not None:
            weight = custom_weight
        else:
            weight = math.hypot(pos1.x()-pos2.x(), pos1.y()-pos2.y())
        
        # 添加边权重文本，负权边显示为红色
        mid_x = (pos1.x() + pos2.x()) / 2
        mid_y = (pos1.y() + pos2.y()) / 2
        weight_text = self.scene.addText(f"{weight:.1f}")
        if weight < 0:
            weight_text.setDefaultTextColor(Qt.red)
        else:
            weight_text.setDefaultTextColor(Qt.blue)
        weight_text.setPos(mid_x - weight_text.boundingRect().width()/2,
                          mid_y - weight_text.boundingRect().height()/2)
        
        self.graph.add_edge(node1, node2, weight=weight)

    def updateEdgeWeight(self, node):
        """更新与指定节点相关的边的权重"""
        # 首先找到用户想要编辑的边
        edges = list(self.graph.out_edges(node, data=True)) + list(self.graph.in_edges(node, data=True))
        
        if not edges:
            QMessageBox.warning(self.parent(), "警告", "该节点没有相连的边!")
            return
            
        # 如果只有一条边，直接让用户输入权重
        if len(edges) == 1:
            u, v, data = edges[0]
            current_weight = data.get('weight', 0)
            
            # 让用户输入新的权重
            new_weight, ok = QInputDialog.getDouble(
                self,
                "编辑边权重",
                f"请为边 {u}->{v} 输入新的权重 (当前: {current_weight:.1f}):",
                value=current_weight,
                decimals=1
            )
            
            if ok:
                self.updateSingleEdgeWeight(u, v, new_weight)
        else:
            # 有多条边，让用户选择要编辑哪条边
            edge_options = []
            edge_data = []
            
            for u, v, data in edges:
                weight = data.get('weight', 0)
                if u == node:  # 出边
                    edge_options.append(f"从 {u} 到 {v} (当前权重: {weight:.1f})")
                else:  # 入边
                    edge_options.append(f"从 {u} 到 {v} (当前权重: {weight:.1f})")
                edge_data.append((u, v, weight))
            
            selected, ok = QInputDialog.getItem(
                self, 
                "选择边", 
                "请选择要编辑的边:",
                edge_options,
                0,  # 默认选择第一项
                False  # 不可编辑
            )
            
            if ok and selected:
                # 解析选择的边
                idx = edge_options.index(selected)
                u, v, current_weight = edge_data[idx]
                
                # 让用户输入新的权重
                new_weight, ok = QInputDialog.getDouble(
                    self,
                    "编辑边权重",
                    f"请为边 {u}->{v} 输入新的权重 (当前: {current_weight:.1f}):",
                    value=current_weight,
                    decimals=1
                )
                
                if ok:
                    self.updateSingleEdgeWeight(u, v, new_weight)
    
    def updateSingleEdgeWeight(self, u, v, new_weight):
        """更新单条边的权重"""
        # 更新图中的边权重
        self.graph[u][v]['weight'] = new_weight
        
        # 移除所有文本项并重新绘制所有边的权重
        # 清除所有文本项
        for item in list(self.scene.items()):
            if isinstance(item, QGraphicsTextItem):
                self.scene.removeItem(item)
        
        # 重新绘制所有边的权重标签
        for edge_u, edge_v, data in self.graph.edges(data=True):
            pos1 = self.node_items[edge_u].pos()
            pos2 = self.node_items[edge_v].pos()
            mid_x = (pos1.x() + pos2.x()) / 2
            mid_y = (pos1.y() + pos2.y()) / 2
            
            weight = data.get('weight', 0)
            weight_text = self.scene.addText(f"{weight:.1f}")
            if weight < 0:
                weight_text.setDefaultTextColor(Qt.red)
            else:
                weight_text.setDefaultTextColor(Qt.blue)
            weight_text.setPos(mid_x - weight_text.boundingRect().width()/2,
                             mid_y - weight_text.boundingRect().height()/2)
        
        # 重新绘制节点编号
        for node_id in self.graph.nodes():
            pos = self.node_items[node_id].pos()
            text = self.scene.addText(str(node_id))
            text.setDefaultTextColor(Qt.black)
            text.setPos(pos.x() - text.boundingRect().width()/2,
                       pos.y() - text.boundingRect().height()/2)
        
        # 检查负权边
        self.checkNegativeWeights()
        
        QMessageBox.information(self.parent(), "成功", f"边 {u}->{v} 的权重已更新为 {new_weight}")

    def mousePressEvent(self, event):
        """根据当前操作模式响应鼠标点击"""
        pos = self.mapToScene(event.pos())
        if self.mode == 'add_node':
            self.addNode(pos)
        elif self.mode == 'add_edge':
            # 判断是否点击到节点
            clicked_node = self.getNodeAtPosition(pos)
            if clicked_node is not None:
                if self.temp_edge_start is None:
                    self.temp_edge_start = clicked_node
                    # 高亮第一个选中的节点
                    self.node_items[clicked_node].setBrush(QBrush(QColor("orange")))
                else:
                    # 第二次点击，添加边
                    if clicked_node != self.temp_edge_start:  # 防止自环
                        # 弹出对话框让用户输入边权重
                        weight, ok = QInputDialog.getDouble(
                            self, 
                            "输入边权重", 
                            "请输入边的权重:",
                            value=0.0,  # 默认值
                            decimals=1   # 小数点后1位
                        )
                        if ok:
                            # 添加边并更新负权边标志
                            self.addEdge(self.temp_edge_start, clicked_node, weight)
                            # 检查是否添加了负权边并通知主窗口
                            self.checkNegativeWeights()
                    # 重置颜色
                    self.node_items[self.temp_edge_start].setBrush(QBrush(QColor("lightblue")))
                    self.temp_edge_start = None
        elif self.mode == 'set_start':
            clicked_node = self.getNodeAtPosition(pos)
            if clicked_node is not None:
                if self.start_node is not None:
                    self.node_items[self.start_node].setBrush(QBrush(QColor("lightblue")))
                self.start_node = clicked_node
                self.node_items[clicked_node].setBrush(QBrush(QColor("green")))
        elif self.mode == 'set_goal':
            clicked_node = self.getNodeAtPosition(pos)
            if clicked_node is not None:
                if self.goal_node is not None:
                    self.node_items[self.goal_node].setBrush(QBrush(QColor("lightblue")))
                self.goal_node = clicked_node
                self.node_items[clicked_node].setBrush(QBrush(QColor("red")))
        elif self.mode == 'edit_weight':
            clicked_node = self.getNodeAtPosition(pos)
            if clicked_node is not None:
                # 先选择边，再输入权重
                self.updateEdgeWeight(clicked_node)
        else:
            super(GraphView, self).mousePressEvent(event)

    def getNodeAtPosition(self, pos):
        """判断 pos 附近是否有节点，若有返回该节点 id"""
        for node_id, item in self.node_items.items():
            item_pos = item.pos()
            if (item_pos - pos).manhattanLength() < 20:  # 距离门限
                return node_id
        return None

    def saveGraph(self, filename):
        """保存当前图结构到文件"""
        data = {
            'nodes': list(self.graph.nodes()),
            'edges': [(u, v, d) for (u, v, d) in self.graph.edges(data=True)],
            'positions': {str(k): (x, y) for k, (x, y) in self.node_positions.items()}
        }
        import json
        with open(filename, 'w') as f:
            json.dump(data, f)

    def loadGraph(self, filename):
        """从文件加载图结构"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)

        # 清除当前图
        self.scene.clear()
        self.graph.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.node_positions.clear()
        
        # 重置节点ID计数器
        self.next_node_id = 0
        
        # 重置起点和终点
        self.start_node = None
        self.goal_node = None
        
        # 创建节点ID映射
        node_id_map = {}

        # 重建节点，保留原始节点ID
        for node_str in data['nodes']:
            node = int(node_str)
            pos = QPointF(*data['positions'][str(node)])
            
            # 手动创建节点并保留原始ID
            radius = 15
            ellipse = QGraphicsEllipseItem(-radius, -radius, 2*radius, 2*radius)
            ellipse.setBrush(QBrush(QColor("lightblue")))
            ellipse.setPen(QPen(Qt.black))
            ellipse.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
            ellipse.setPos(pos)
            self.scene.addItem(ellipse)
            
            # 添加节点编号文本
            text = self.scene.addText(str(node))
            text.setDefaultTextColor(Qt.black)
            text.setPos(pos.x() - text.boundingRect().width()/2,
                       pos.y() - text.boundingRect().height()/2)
            
            # 保存节点信息
            self.node_items[node] = ellipse
            self.node_positions[node] = (pos.x(), pos.y())
            self.graph.add_node(node)
            
            # 更新最大节点ID
            if node >= self.next_node_id:
                self.next_node_id = node + 1

        # 检查是否包含负权边的标志
        has_negative_weights = False
        
        # 重建边
        for u_str, v_str, d in data['edges']:
            u, v = int(u_str), int(v_str)
            
            # 尝试恢复边权重
            weight = None
            if 'weight' in d:
                weight = d['weight']
                # 检查是否是负权边
                if weight < 0:
                    has_negative_weights = True
                
            # 添加边
            self.addEdge(u, v, weight)
        
        # 设置图是否包含负权边的标志
        self.has_negative_weights = has_negative_weights
        
        # 通知主窗口图中是否包含负权边
        if self.parent():
            self.parent().has_negative_weights = has_negative_weights
            
            # 根据是否包含负权边启用或禁用相应算法
            if has_negative_weights:
                self.parent().disableUnsupportedAlgorithms()
                QMessageBox.information(self.parent(), "提示", "加载的图包含负权边，只能使用SPFA算法")
            else:
                self.parent().enableAllAlgorithms()
        
        # 清除当前起点和终点标记
        self.start_node = None
        self.goal_node = None

    def create_example_graph(self, with_negative_weights=False):
        """创建一个包含30个随机分布节点的有向连通图，边数量适中以保持清晰
        
        参数:
            with_negative_weights: 是否包含负权边，默认为False
        """
        import random
        
        # 清除现有图
        self.scene.clear()
        self.graph.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.node_positions.clear()
        self.next_node_id = 0
        
        # 重置起点和终点
        self.start_node = None
        self.goal_node = None
        
        # 记录图是否包含负权边
        self.has_negative_weights = with_negative_weights
        
        # 使用网格布局生成30个均匀分布的节点
        width = 800
        height = 600
        margin = 50
        nodes = []
        
        # 计算网格大小和节点间距
        grid_cols = 6  # 水平方向6个节点
        grid_rows = 5  # 垂直方向5个节点
        cell_width = (width - 2 * margin) / (grid_cols - 1)
        cell_height = (height - 2 * margin) / (grid_rows - 1)
        
        # 在网格点上添加节点，并添加一些随机偏移以避免完全规则排列
        for row in range(grid_rows):
            for col in range(grid_cols):
                # 跳过一些位置以得到恰好30个节点
                if row == grid_rows - 1 and col >= grid_cols - 0:  # 最后一行只放5个节点
                    continue
                    
                # 基础位置
                base_x = margin + col * cell_width
                base_y = margin + row * cell_height
                
                # 添加小的随机偏移（不超过单元格大小的20%）
                offset_x = random.uniform(-0.2 * cell_width, 0.2 * cell_width)
                offset_y = random.uniform(-0.2 * cell_height, 0.2 * cell_height)
                
                x = base_x + offset_x
                y = base_y + offset_y
                
                nodes.append(self.addNode(QPointF(x, y)))
        
        # 生成有向边，确保图是强连通的
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    pos1 = self.node_items[nodes[i]].pos()
                    pos2 = self.node_items[nodes[j]].pos()
                    weight = math.hypot(pos1.x() - pos2.x(), pos1.y() - pos2.y())
                    edges.append((weight, nodes[i], nodes[j]))
        edges.sort()  # 按权重（距离）排序
        
        # 首先确保图是强连通的
        # 创建一个环，确保每个节点都可以到达其他节点
        for i in range(len(nodes)):
            next_node = nodes[(i + 1) % len(nodes)]
            self.addEdge(nodes[i], next_node)
        
        # 随机添加额外的边，但数量较少以保持图的美观
        if with_negative_weights:
            # 每个节点平均添加0.5条额外的出边（减少边的数量）
            extra_edges = len(nodes) // 2
            added_extra = 0
            random.shuffle(edges)  # 随机打乱边的顺序
            
            # 用于检测添加边后是否会形成负权回路的函数
            def will_form_negative_cycle(u, v, weight):
                # 如果权重不是负数，不会形成负权回路
                if weight >= 0:
                    return False
                
                # 检查是否已经存在从v到u的路径
                # 如果存在，并且路径长度小于|weight|，则会形成负权回路
                
                # 使用Dijkstra算法查找从v到u的最短路径
                # 注意：Dijkstra不适用于负权图，但我们只是在添加负权边之前检查
                try:
                    path_length = nx.dijkstra_path_length(self.graph, v, u, weight='weight')
                    # 如果从v到u的路径长度小于|weight|，则会形成负权回路
                    if path_length < abs(weight):
                        return True
                except nx.NetworkXNoPath:
                    # 如果没有从v到u的路径，则不会形成负权回路
                    pass
                
                # 特别检查两个节点之间的小回路
                if self.graph.has_edge(v, u):
                    v_to_u_weight = self.graph[v][u]['weight']
                    # 如果v->u + u->v(新边) < 0，则形成负权回路
                    if v_to_u_weight + weight < 0:
                        return True
                
                return False
            
            for weight, u, v in edges:
                if added_extra >= extra_edges:
                    break
                if not self.graph.has_edge(u, v):
                    # 随机决定是否为负权边
                    if random.random() < 0.4:  # 40%的概率是负权边
                        neg_weight = -random.uniform(1, 5)  # 负权值在-1到-5之间
                        
                        # 检查添加这条负权边是否会形成负权回路
                        if not will_form_negative_cycle(u, v, neg_weight):
                            self.addEdge(u, v, neg_weight)
                            added_extra += 1
                        else:
                            # 如果会形成负权回路，则添加正权边代替
                            self.addEdge(u, v)
                            added_extra += 1
                    else:
                        self.addEdge(u, v)
                        added_extra += 1
        else:
            # 每个节点平均添加0.5条额外的出边（减少边的数量）
            extra_edges = len(nodes) // 2
            added_extra = 0
            
            # 优先添加距离较近的边
            for weight, u, v in edges:
                if added_extra >= extra_edges:
                    break
                # 避免添加距离过远的边（超过平均单元格大小的2倍）
                if weight > 2 * (cell_width + cell_height) / 2:
                    continue
                if not self.graph.has_edge(u, v):
                    self.addEdge(u, v)
                    added_extra += 1

    def animateSearch(self, steps, final_path, elapsed_time):
        """利用定时器逐步展现搜索步骤，搜索步骤存储在 steps 列表中"""
        self.search_steps = steps
        self.final_path = final_path
        self.current_step_index = 0
        self.clearSelectionColors()
        
        # 计算每个步骤的动画时间
        default_step_time = 500  # 默认每个步骤0.5秒
        total_default_time = len(steps) * default_step_time / 1000  # 默认总时间（秒）
        
        if total_default_time > 10.0:
            # 如果默认总时间超过10秒，将10秒平均分配到每个步骤
            step_interval = (10.0 * 1000) / len(steps)
        else:
            # 如果默认总时间不超过10秒，每个步骤保持0.5秒
            step_interval = default_step_time
        
        self.timer.start(int(step_interval))  # 设置定时器间隔

    def animateStep(self):
        """在定时器中调用，依次展示步骤"""
        if self.current_step_index < len(self.search_steps):
            step = self.search_steps[self.current_step_index]
            step_type = step[0]
            
            if step_type == 'expand':
                # 更新节点颜色：只有当前访问的节点为黄色，起点为绿色，终点为红色
                for node_id, item in self.node_items.items():
                    if node_id == self.start_node:
                        item.setBrush(QBrush(QColor("green")))
                    elif node_id == self.goal_node:
                        # 如果目标节点是当前扩展的节点，标记为黄色
                        if node_id == step[1]:
                            item.setBrush(QBrush(QColor("yellow")))
                        else:
                            item.setBrush(QBrush(QColor("red")))
                    elif node_id == step[1]:
                        # 只有当前扩展的节点标记为黄色
                        item.setBrush(QBrush(QColor("yellow")))
                    else:
                        item.setBrush(QBrush(QColor("lightblue")))
            elif step_type == 'backtrack':
                # 回溯时保持节点颜色不变
                pass
            elif step_type == 'enqueue':
                # 入队时不改变节点颜色
                pass
            elif step_type == 'update':
                # 更新最佳路径时不改变节点颜色
                pass
            elif step_type == 'stats':
                cost, expanded_count, elapsed_time = step[1:]
                QMessageBox.information(self.parent(), "搜索结果统计",
                    f"路径长度：{cost:.2f}\n"
                    f"扩展节点数：{expanded_count}\n"
                    f"执行时间：{elapsed_time*1000:.2f}ms")
            self.current_step_index += 1
        else:
            self.timer.stop()
            # 搜索结束后，显示最终路径
            for node_id, item in self.node_items.items():
                if node_id == self.start_node:
                    item.setBrush(QBrush(QColor("green")))
                elif node_id == self.goal_node:
                    item.setBrush(QBrush(QColor("red")))
                elif self.final_path and node_id in self.final_path:
                    item.setBrush(QBrush(QColor("red")))
                else:
                    item.setBrush(QBrush(QColor("lightblue")))

    def checkNegativeWeights(self):
        """检查图中是否存在负权边，并更新标志"""
        has_negative = False
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 0) < 0:
                has_negative = True
                break
        
        # 更新负权边标志
        self.has_negative_weights = has_negative
        
        # 通知主窗口
        if self.parent():
            # 如果状态改变，更新主窗口中的算法可用性
            if not hasattr(self.parent(), 'has_negative_weights') or self.parent().has_negative_weights != has_negative:
                self.parent().has_negative_weights = has_negative
                if has_negative:
                    self.parent().disableUnsupportedAlgorithms()
                    QMessageBox.information(self.parent(), "提示", "图中存在负权边，只能使用SPFA算法")
                else:
                    self.parent().enableAllAlgorithms()

##############################################
# 主窗口：包含菜单栏和 GraphView
##############################################
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("最短路径搜索可视化")
        self.graphView = GraphView(self)
        self.setCentralWidget(self.graphView)
        
        # 默认选择 BFS 算法
        self.selected_algorithm = "BFS"
        
        # 记录图是否包含负权边
        self.has_negative_weights = False
        
        # 创建工具栏
        self.createToolBar()
        self.createMenus()

    def createToolBar(self):
        # 创建工具栏
        toolbar = self.addToolBar("主工具栏")
        
        # 添加"开始搜索"按钮到工具栏
        searchAction = QAction("开始搜索", self)
        searchAction.triggered.connect(self.startSearch)
        toolbar.addAction(searchAction)
        
        # 添加操作模式按钮到工具栏
        addNodeAction = QAction("添加节点", self)
        addNodeAction.triggered.connect(lambda: self.graphView.setMode('add_node'))
        toolbar.addAction(addNodeAction)
        
        addEdgeAction = QAction("添加边", self)
        addEdgeAction.triggered.connect(lambda: self.graphView.setMode('add_edge'))
        toolbar.addAction(addEdgeAction)
        
        # 添加编辑边权重模式
        editWeightAction = QAction("编辑边权重", self)
        editWeightAction.triggered.connect(lambda: self.graphView.setMode('edit_weight'))
        toolbar.addAction(editWeightAction)
        
        setStartAction = QAction("设置起点", self)
        setStartAction.triggered.connect(lambda: self.graphView.setMode('set_start'))
        toolbar.addAction(setStartAction)
        
        setGoalAction = QAction("设置终点", self)
        setGoalAction.triggered.connect(lambda: self.graphView.setMode('set_goal'))
        toolbar.addAction(setGoalAction)

    def createMenus(self):
        menubar = self.menuBar()
        
        # 操作菜单：添加节点、添加边、设置起点、设置终点
        modeMenu = menubar.addMenu("操作")
        addNodeAction = QAction("添加节点", self)
        addNodeAction.triggered.connect(lambda: self.graphView.setMode('add_node'))
        modeMenu.addAction(addNodeAction)
        
        addEdgeAction = QAction("添加边", self)
        addEdgeAction.triggered.connect(lambda: self.graphView.setMode('add_edge'))
        modeMenu.addAction(addEdgeAction)
        
        # 添加编辑边权重选项到菜单
        editWeightAction = QAction("编辑边权重", self)
        editWeightAction.triggered.connect(lambda: self.graphView.setMode('edit_weight'))
        modeMenu.addAction(editWeightAction)
        
        setStartAction = QAction("设置起点", self)
        setStartAction.triggered.connect(lambda: self.graphView.setMode('set_start'))
        modeMenu.addAction(setStartAction)
        
        setGoalAction = QAction("设置终点", self)
        setGoalAction.triggered.connect(lambda: self.graphView.setMode('set_goal'))
        modeMenu.addAction(setGoalAction)
        
        # 算法选择菜单
        self.algoMenu = menubar.addMenu("算法选择")
        dfsAction = QAction("DFS", self)
        dfsAction.triggered.connect(lambda: self.setAlgorithm("DFS"))
        self.algoMenu.addAction(dfsAction)
        
        bfsAction = QAction("BFS", self)
        bfsAction.triggered.connect(lambda: self.setAlgorithm("BFS"))
        self.algoMenu.addAction(bfsAction)
        
        astarAction = QAction("A*", self)
        astarAction.triggered.connect(lambda: self.setAlgorithm("A*"))
        self.algoMenu.addAction(astarAction)
        
        dijkstraAction = QAction("Dijkstra", self)
        dijkstraAction.triggered.connect(lambda: self.setAlgorithm("Dijkstra"))
        self.algoMenu.addAction(dijkstraAction)
        
        spfaAction = QAction("SPFA", self)
        spfaAction.triggered.connect(lambda: self.setAlgorithm("SPFA"))
        self.algoMenu.addAction(spfaAction)
        
        # 添加随机生成图的按钮
        generateAction = QAction("随机生成图（无负权边）", self)
        generateAction.triggered.connect(self.generateNormalGraph)
        modeMenu.addAction(generateAction)
        
        # 添加生成带负权边的图的按钮
        generateNegativeAction = QAction("随机生成图（带负权边）", self)
        generateNegativeAction.triggered.connect(self.generateNegativeGraph)
        modeMenu.addAction(generateNegativeAction)
        
        # "开始搜索"动作（保留在菜单中，同时也在工具栏中显示）
        searchAction = QAction("开始搜索", self)
        searchAction.triggered.connect(self.startSearch)
        self.algoMenu.addAction(searchAction)
        
        # 文件菜单
        fileMenu = menubar.addMenu("文件")
        
        # 保存图结构
        saveAction = QAction("保存图结构", self)
        saveAction.triggered.connect(self.saveGraph)
        fileMenu.addAction(saveAction)
        
        # 加载图结构
        loadAction = QAction("加载图结构", self)
        loadAction.triggered.connect(self.loadGraph)
        fileMenu.addAction(loadAction)

    def generateNormalGraph(self):
        """生成不包含负权边的图"""
        self.graphView.create_example_graph(with_negative_weights=False)
        self.has_negative_weights = False
        # 重置算法选择
        self.enableAllAlgorithms()
        QMessageBox.information(self, "图生成", "已生成新图，起点和终点已重置")
    
    def generateNegativeGraph(self):
        """生成包含负权边的图"""
        self.graphView.create_example_graph(with_negative_weights=True)
        self.has_negative_weights = True
        # 禁用不支持负权边的算法
        self.disableUnsupportedAlgorithms()
        QMessageBox.information(self, "图生成", "已生成包含负权边的图，起点和终点已重置\n注意：图中不存在负权环")
    
    def enableAllAlgorithms(self):
        """启用所有算法"""
        for action in self.algoMenu.actions():
            if action.text() in ["DFS", "BFS", "A*", "Dijkstra", "SPFA"]:
                action.setEnabled(True)
        # 默认选择BFS算法
        self.setAlgorithm("BFS")
    
    def disableUnsupportedAlgorithms(self):
        """禁用不支持负权边的算法"""
        for action in self.algoMenu.actions():
            if action.text() in ["DFS", "BFS", "A*", "Dijkstra"]:
                action.setEnabled(False)
            elif action.text() == "SPFA":
                action.setEnabled(True)
        # 默认选择SPFA算法
        self.setAlgorithm("SPFA")
    
    def setAlgorithm(self, algo):
        self.selected_algorithm = algo
        # 弹窗提示用户已选择的算法
        QMessageBox.information(self, "算法选择", f"已选择{algo}算法")
        # 重置节点颜色，只保留起点和终点的标记
        for node_id, item in self.graphView.node_items.items():
            if node_id == self.graphView.start_node:
                item.setBrush(QBrush(QColor("green")))
            elif node_id == self.graphView.goal_node:
                item.setBrush(QBrush(QColor("red")))
            else:
                item.setBrush(QBrush(QColor("lightblue")))

    def saveGraph(self):
        filename, _ = QFileDialog.getSaveFileName(self, "保存图结构", "", "JSON Files (*.json)")
        if filename:
            self.graphView.saveGraph(filename)
            
    def loadGraph(self):
        filename, _ = QFileDialog.getOpenFileName(self, "加载图结构", "", "JSON Files (*.json)")
        if filename:
            self.graphView.loadGraph(filename)

    def startSearch(self):
        """检查起点和终点，依据选择的算法调用对应的搜索函数，并启动动画展示搜索过程"""
        if self.graphView.start_node is None or self.graphView.goal_node is None:
            QMessageBox.warning(self, "警告", "请先设置起点和终点！")
            return
        
        # 检查起点和终点是否在当前图中
        if (self.graphView.start_node not in self.graphView.graph.nodes() or 
            self.graphView.goal_node not in self.graphView.graph.nodes()):
            QMessageBox.warning(self, "警告", "起点或终点不在当前图中！\n请先重新设置起点和终点。")
            self.graphView.start_node = None
            self.graphView.goal_node = None
            return
            
        if not hasattr(self, 'selected_algorithm') or self.selected_algorithm is None:
            QMessageBox.warning(self, "警告", "请先选择搜索算法！")
            return
        
        # 重置所有节点颜色为默认（浅蓝色），只保留起点和终点的标记
        for node_id, item in self.graphView.node_items.items():
            if node_id == self.graphView.start_node:
                item.setBrush(QBrush(QColor("green")))
            elif node_id == self.graphView.goal_node:
                item.setBrush(QBrush(QColor("red")))
            else:
                item.setBrush(QBrush(QColor("lightblue")))
        
        algo = self.selected_algorithm
        G = self.graphView.graph
        start = self.graphView.start_node
        goal = self.graphView.goal_node
        
        # 执行选择的算法
        if algo == "BFS":
            path, steps = self.bfs_steps(G, start, goal)
        elif algo == "DFS":
            path, steps = self.dfs_steps(G, start, goal)
        elif algo == "A*":
            path, steps = self.astar_steps(G, start, goal, self.graphView.node_positions)
        elif algo == "Dijkstra":
            path, steps = self.dijkstra_steps(G, start, goal)
        elif algo == "SPFA":
            # 如果图包含负权边，不启用剪枝优化
            use_pruning = not self.has_negative_weights
            path, steps = self.spfa_steps(G, start, goal, use_pruning)
        else:
            QMessageBox.warning(self, "错误", "未知算法！")
            return
        
        # 显示搜索结果
        if path is None:
            QMessageBox.information(self, "结果", "未找到路径！")
        else:
            QMessageBox.information(self, "结果", f"找到路径：{path}\n路径长度：{steps[-1][1]:.2f}")
        
        # 启动动画展示搜索过程，传递搜索耗时
        elapsed_time = steps[-1][3] if steps and len(steps[-1]) > 3 else 1.0
        self.graphView.animateSearch(steps, path, elapsed_time)

    ##############################################
    # 以下为各算法带步骤记录的实现（记录"expand"、"enqueue"等步骤，用于动画显示）
    ##############################################
    def bfs_steps(self, G, start, goal):
        queue = deque([(start, [start], 0)])  # 添加路径长度统计
        visited = {start: 0}  # 记录到达每个节点的最小代价
        steps = []
        expanded_count = 0
        start_time = time.time()
        best_path = None
        best_cost = float('inf')
        
        while queue:
            current, path, cost = queue.popleft()
            expanded_count += 1
            steps.append(('expand', current, path.copy(), cost))
            
            # 如果当前成本已经超过最佳成本，则跳过（剪枝优化）
            if cost > best_cost:
                continue
                
            if current == goal:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
                    steps.append(('update', current, path.copy(), cost))
                continue  # 继续搜索，可能存在更短的路径
                
            for neighbor in G.neighbors(current):
                new_cost = cost + G[current][neighbor]['weight']
                # 剪枝：如果新路径成本已经超过最佳成本，则跳过
                if new_cost >= best_cost:
                    continue
                # 如果找到更优的路径或者是第一次访问该节点
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    queue.append((neighbor, path + [neighbor], new_cost))
                    steps.append(('enqueue', neighbor, path + [neighbor], new_cost))
        
        elapsed_time = time.time() - start_time
        # 修正统计信息中的成本值
        if best_path:
            steps.append(('stats', best_cost, expanded_count, elapsed_time))
        else:
            steps.append(('stats', float('inf'), expanded_count, elapsed_time))
        # 返回找到的最佳路径，而不是None
        return best_path, steps

    def dfs_steps(self, G, start, goal):
        steps = []
        best_path = None
        best_cost = float('inf')
        expanded_count = 0
        start_time = time.time()
        
        def dfs_recursive(current, path, visited, cost):
            nonlocal best_path, best_cost, expanded_count
            expanded_count += 1
            steps.append(('expand', current, path.copy(), cost))
            if current == goal:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
                    steps.append(('update', current, path.copy(), cost))
                return
            # 按照距离目标节点的远近对邻居节点排序
            neighbors = list(G.neighbors(current))
            neighbors.sort(key=lambda n: math.hypot(
                self.graphView.node_positions[n][0] - self.graphView.node_positions[goal][0],
                self.graphView.node_positions[n][1] - self.graphView.node_positions[goal][1]
            ))
            for neighbor in neighbors:
                if neighbor not in visited:
                    w = G[current][neighbor]['weight']
                    if cost + w < best_cost:  # 剪枝：如果当前路径已经超过最优解，则不再继续
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs_recursive(neighbor, path, visited, cost + w)
                        # 回溯：记录节点的回溯过程
                        steps.append(('backtrack', neighbor, path.copy(), cost + w))
                        path.pop()
                        visited.remove(neighbor)
            
        visited = {start}
        dfs_recursive(start, [start], visited, 0)
        elapsed_time = time.time() - start_time
        steps.append(('stats', best_cost, expanded_count, elapsed_time))
        return best_path, steps

    def astar_steps(self, G, start, goal, positions):
        def heuristic(n1, n2):
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            # 使用欧几里得距离作为启发式函数，更准确地估计实际距离
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)
        
        steps = []
        expanded_count = 0
        start_time = time.time()
        
        heap = [(heuristic(start, goal), 0, start, [start])]
        visited = {}
        best_path = None
        best_cost = float('inf')
        
        while heap:
            f, cost, current, path = heapq.heappop(heap)
            expanded_count += 1
            steps.append(('expand', current, path.copy(), cost))
            
            # 如果当前成本已经超过最佳成本，则跳过（剪枝优化）
            if cost > best_cost:
                continue
                
            if current == goal:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
                    steps.append(('update', current, path.copy(), cost))
                continue  # 继续搜索，可能存在更短的路径
                
            if current in visited and visited[current] <= cost:
                continue
                
            visited[current] = cost
            neighbors = list(G.neighbors(current))
            # 按启发式值对邻居节点排序
            neighbors.sort(key=lambda n: heuristic(n, goal))
            
            for neighbor in neighbors:
                w = G[current][neighbor]['weight']
                new_cost = cost + w
                # 剪枝：如果新路径成本已经超过最佳成本，则跳过
                if new_cost >= best_cost:
                    continue
                new_f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(heap, (new_f, new_cost, neighbor, path + [neighbor]))
                steps.append(('enqueue', neighbor, path + [neighbor], new_cost))
        
        elapsed_time = time.time() - start_time
        # 修正统计信息中的成本值
        if best_path:
            steps.append(('stats', best_cost, expanded_count, elapsed_time))
        else:
            steps.append(('stats', float('inf'), expanded_count, elapsed_time))
        # 返回找到的最佳路径，而不是None
        return best_path, steps

    def dijkstra_steps(self, G, start, goal):
        steps = []
        expanded_count = 0
        start_time = time.time()
        
        # 初始化距离字典，所有节点距离设为无穷大，起点设为0
        dist = {node: float('inf') for node in G.nodes()}
        dist[start] = 0
        
        # 初始化优先队列，元素为 (距离, 节点, 路径)
        heap = [(0, start, [start])]
        # 记录已经确定最短路径的节点
        visited = set()
        
        while heap:
            cost, current, path = heapq.heappop(heap)
            
            # 如果节点已经处理过，跳过
            if current in visited:
                continue
                
            # 标记当前节点为已访问
            visited.add(current)
            expanded_count += 1
            steps.append(('expand', current, path.copy(), cost))
            
            # 如果到达目标节点，结束搜索
            if current == goal:
                elapsed_time = time.time() - start_time
                steps.append(('stats', cost, expanded_count, elapsed_time))
                return path, steps
            
            # 遍历当前节点的所有邻居
            for neighbor in G.neighbors(current):
                if neighbor in visited:
                    continue
                    
                # 计算通过当前节点到达邻居的新距离
                weight = G[current][neighbor]['weight']
                new_cost = cost + weight
                
                # 如果找到更短的路径，更新距离并加入队列
                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_cost, neighbor, new_path))
                    steps.append(('enqueue', neighbor, new_path, new_cost))
        
        # 如果无法到达目标节点
        elapsed_time = time.time() - start_time
        steps.append(('stats', float('inf'), expanded_count, elapsed_time))
        return None, steps
        
    def spfa_steps(self, G, start, goal, use_pruning=True):
        """SPFA算法实现，带步骤记录
        
        参数:
            G: 图
            start: 起点
            goal: 终点
            use_pruning: 是否启用剪枝优化，对于带负权边的图应设为False
        """
        steps = []
        expanded_count = 0
        start_time = time.time()
        
        # 初始化距离字典，所有节点距离设为无穷大，起点设为0
        dist = {node: float('inf') for node in G.nodes()}
        dist[start] = 0
        
        # 初始化队列和路径记录
        queue = deque([(start, [start])])
        in_queue = {node: False for node in G.nodes()}
        in_queue[start] = True
        
        # 记录每个节点的最佳路径
        paths = {start: [start]}
        
        # 记录最佳路径和成本
        best_path = None
        best_cost = float('inf')
        
        # 由于已确保图中不存在负权环，不再需要记录入队次数
        
        while queue:
            current, path = queue.popleft()
            in_queue[current] = False
            expanded_count += 1
            steps.append(('expand', current, path.copy(), dist[current]))
            
            # 如果到达目标节点，记录最佳路径，但不立即结束
            if current == goal and dist[current] < best_cost:
                best_cost = dist[current]
                best_path = path.copy()
                steps.append(('update', current, path.copy(), dist[current]))
                
                # 如果启用剪枝且不存在负权边，可以提前结束
                if use_pruning and not hasattr(self.graphView, 'has_negative_weights') or not self.graphView.has_negative_weights:
                    break
            
            # 遍历当前节点的所有邻居
            for neighbor in G.neighbors(current):
                weight = G[current][neighbor]['weight']
                new_cost = dist[current] + weight
                
                # 如果找到更短的路径，更新距离
                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    new_path = path + [neighbor]
                    paths[neighbor] = new_path
                    steps.append(('update', neighbor, new_path, new_cost))
                    
                    # 如果邻居节点不在队列中，将其加入队列（已移除入队次数限制）
                    if not in_queue[neighbor]:
                        queue.append((neighbor, new_path))
                        in_queue[neighbor] = True
                        steps.append(('enqueue', neighbor, new_path, new_cost))
        
        # 搜索结束后，检查是否找到到达目标的路径
        elapsed_time = time.time() - start_time
        if goal in paths:
            steps.append(('stats', dist[goal], expanded_count, elapsed_time))
            return paths[goal], steps
        else:
            steps.append(('stats', float('inf'), expanded_count, elapsed_time))
            return None, steps

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())