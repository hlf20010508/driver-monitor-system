from torch_geometric_temporal.nn import stgcn
import torch.nn as nn
import torch.nn.functional as F
import torch

class STGCN(nn.Module):
    def __init__(self, num_nodes, time_len, class_num, size=3, K=3):
        super(STGCN, self).__init__()
        self.conv1 = stgcn.STConv(num_nodes=num_nodes, in_channels=2, hidden_channels=16, out_channels=64, kernel_size=size, K=K)
        self.conv2 = stgcn.STConv(num_nodes=num_nodes, in_channels=64, hidden_channels=16, out_channels=64, kernel_size=size, K=K)
        self.fc = nn.Linear(64 * (time_len - 8) * num_nodes, class_num)
    def forward(self, x, edge_index):# x(batch_size, seq_len, num_nodes, in_channels)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
