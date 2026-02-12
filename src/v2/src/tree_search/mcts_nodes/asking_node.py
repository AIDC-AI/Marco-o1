from tree_search.mcts_nodes.base_node import BaseNode


class AskingNode(BaseNode):
    """
    评估节点，用于评估
    """
    node_action_name = 'asking'
    node_action_description = ['<asking>', '</asking>']

    def __init__(self, parent):
        super().__init__(parent)

    def post_init(self, ask_response):
        if len(self.all_path_value) == 0:
            ask_response = '?'
        INPUT_PATTERN = '%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n'
        self.all_path_value = INPUT_PATTERN % (self.all_path_value, ask_response)