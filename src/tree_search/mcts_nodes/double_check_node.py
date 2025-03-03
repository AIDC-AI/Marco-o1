from tree_search.mcts_nodes.base_node import BaseNode


class DoubleCheckNode(BaseNode):
    """

    """
    node_action_name = 'double_check'
    node_action_description = ['<double-check>', '</double-check>']

    def __init__(self, parent):
        super().__init__(parent)
