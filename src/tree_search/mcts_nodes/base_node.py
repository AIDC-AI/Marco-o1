from collections import defaultdict


class NodeCounter:
    _id_counter = 0

    @classmethod
    def get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def set_to_zero(cls):
        cls._id_counter = 0


class BaseNode():
    """
    基础节点，是所有特殊节点的父类
    """
    node_action_name = 'base'  # 节点动作名称,日志中使用
    node_action_description = ''  # 节点动作描述，暂时没用
    prefill_text = []  # 预填充文本，用于生成时预填充，引导模型动作
    show_in_history = True  # 是否在history中显示prefill
    use_special_model = False  # 是否使用第二个特殊模型（需要api支持）

    def __init__(self, parent):
        self.node_value = ''
        self.all_path_value = ''
        self.is_terminal = False
        self.is_fully_expanded = False
        self.user_question = ''
        self.parent = parent
        self.num_visits = 0  # N
        self.total_reward = 0  # V
        self.children = defaultdict(list)
        self.total_children_num = 0
        self.other_info = {}
        self.children_ids = []

        if type(self) == BaseNode:
            NodeCounter.set_to_zero()

        self.node_id = NodeCounter.get_next_id()

    def addChild(self, action, childNode):
        """添加子节点"""
        self.children[action].append(childNode)
        self.total_children_num += 1
        self.children_ids.append(childNode.node_id)

    def getChildren(self):
        """获取所有子节点"""
        total_children = []
        for key, children in self.children.items():
            total_children.extend(children)
        return total_children

    def find_parent_by_action(self, action):
        """获取指定动作的父节点"""
        node = self.parent
        while node.parent is not None:
            if node.node_action_name == action:
                return node
            node = node.parent
        return None

    def getDepth(self):
        """获取节点深度"""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def __repr__(self):
        return f'node_id: {self.node_id}, node_action_name: {self.node_action_name}, total_reward: {self.total_reward}, num_visits: {self.num_visits}, total_children_num: {self.total_children_num}, depth: {self.getDepth()}, is_terminal: {self.is_terminal}\n'

    def __str__(self):
        return f'[node_id: {self.node_id}, node_action_name: {self.node_action_name}, total_reward: {self.total_reward}, num_visits: {self.num_visits}, total_children_num: {self.total_children_num}, depth: {self.getDepth()}, is_terminal: {self.is_terminal}]:\t'

    def get_json_dict(self):
        return {
            "node_id": self.node_id,
            "node_action_name": self.node_action_name,
            'total_reward': self.total_reward,
            'num_visits': self.num_visits,
            'total_children_num': self.total_children_num,
            "parent_id": self.parent.node_id if self.parent is not None else None,
            'all_path_value': self.all_path_value,
            'is_terminal': self.is_terminal,
            'node_value': self.node_value,
            "children_ids": self.children_ids,
            'info': self.other_info,
        }
