from graphSage import SageConv
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from coarsening import coarseningGraph


class AMGpool(nn.Module):
    def __init__(self):
        super(AMGpool, self).__init__()

    def forward(self, x, adj, mask=None, log=False):

        xnext, anext = coarseningGraph(x, adj)

        return xnext, anext
