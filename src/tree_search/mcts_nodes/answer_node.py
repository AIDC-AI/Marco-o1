from tree_search.mcts_nodes.base_node import BaseNode


class AnswerNode(BaseNode):
    """
    答案节点，用于生成答案
    """
    node_action_name = 'answer'
    node_action_description = ['<answer>', '</answer>']

    def __init__(self, parent):
        super().__init__(parent)
