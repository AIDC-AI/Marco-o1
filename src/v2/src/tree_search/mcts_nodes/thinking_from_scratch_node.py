from tree_search.mcts_nodes.base_node import BaseNode


class ThinkingFromScratchNode(BaseNode):
    """
    """
    node_action_name = 'thinking_from_scratch'
    node_action_description = ['<thinking_from_scratch>', '</thinking_from_scratch>']

    def __init__(self, parent):
        super().__init__(parent)
