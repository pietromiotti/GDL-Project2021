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
        #z_l = self.Z(x, adj)
        #s_l = F.softmax(self.S(x, adj), dim=-1)

        #xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        #anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        return xnext, anext
