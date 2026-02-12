from tree_search.mcts_nodes.base_node import BaseNode


class ReflectionNode(BaseNode):
    """
    反思节点，用于反思
    """
    node_action_name = 'reflection'
    node_action_description = ['<reflection>', '</reflection>']

    def __init__(self, parent):
        super().__init__(parent)
