from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_geometric.utils import *
from torch import Tensor
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.utils as pyg_utils
from torch_scatter import scatter_add


class SageConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SageConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    #IMPLEMENTED THE FORWARD STEP:
    #X' = NORM(RELU(WEIGHT *(AGGREGATION(D^(-0.5) A D^(-0.5) X))))

    '''
    Implementation choiced (Model Decisions):
    - Normalization of Adjancecy Matrix
    - Add self loop
    '''
    def forward(self, x, edge_index):

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        #Aggregation
        aggregated = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

        #Neural Network
        out = self.lin(aggregated)

        #Relu Activation Function
        out = F.relu(out)

        #normalization according to Graph Sage
        out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, out_channels]
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out

'''
class SageConv(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SageConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = Linear(in_channels, out_channels)


    def forward(self, x, edge_index):

        #SelfLoop
        edge_index, _ =

        #Normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]

        #Propagation/Aggregation and normalization with normalized Weight matrix
        out = self.propagate(edge_index, x=x, norm=norm)

        #Multiplied by W and then
        out = self.linear(out)

        #Apply the Relu-Activation Function
        out = F.relu(out)

        return out
'''
