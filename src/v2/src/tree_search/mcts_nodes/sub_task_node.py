from tree_search.mcts_nodes.base_node import BaseNode


class SubTaskNode(BaseNode):
    """
    子任务节点，用于生成子任务
    """
    node_action_name = 'sub_task'
    node_action_description = ['<sub-task>', '</sub-task>']
    prefill_text = []

    def __init__(self, parent):
        super().__init__(parent)
