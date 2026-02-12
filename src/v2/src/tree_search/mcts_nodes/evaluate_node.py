from tree_search.mcts_nodes.base_node import BaseNode


class EvaluateNode(BaseNode):
    """
    评估节点，用于评估
    """
    node_action_name = 'evaluate'
    node_action_description = ['<evaluate>', '</evaluate>']

    def __init__(self, parent):
        super().__init__(parent)
