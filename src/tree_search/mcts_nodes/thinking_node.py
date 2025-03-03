from tree_search.mcts_nodes.base_node import BaseNode


class ThinkingNode(BaseNode):
    """
    思考节点，用于思考
    """
    node_action_name = 'thinking'
    node_action_description = ['<thinking>', '</thinking>']

    def __init__(self, parent):
        super().__init__(parent)