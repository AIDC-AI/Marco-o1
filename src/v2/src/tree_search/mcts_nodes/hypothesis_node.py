from tree_search.mcts_nodes.base_node import BaseNode


class HypothesisNode(BaseNode):
    """
    假设节点，用于生成假设
    """
    node_action_name = 'hypothesis'
    node_action_description = ['<hypothesis>', '</hypothesis>']

    def __init__(self, parent):
        super().__init__(parent)
