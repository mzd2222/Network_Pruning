import torch
import numpy as np
from torch import nn

class channel_selection(nn.Module):
    """
    从BN层的输出中选择通道。它应该直接放在BN层之后，此层的输出形状由self.indexes中的1的个数决定
    """
    def __init__(self, num_channels):
        """
        使用长度和通道数相同的全1向量初始化"indexes", 剪枝过程中，将要剪枝的通道对应的indexes位置设为0
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        参数：
        输入Tensor维度: (N,C,H,W)，这也是BN层的输出Tensor
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output
