import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime
from coarsening import coarseningGraph
import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from graphSage import SageConv
from diffPool import diffPool

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

'''
Model with DiffPool
'''

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='graph'):
        super(GNNStack, self).__init__()
        #Define task
        self.task = task
        #Init the list of layers
        self.net = nn.ModuleList(
            [SageConv(input_dim, hidden_dim),
             SageConv(hidden_dim, hidden_dim),
             diffPool(hidden_dim, 30, hidden_dim),
             SageConv(hidden_dim, hidden_dim),
             SageConv(hidden_dim, hidden_dim),
             ]
        )

        #Layers Normalization
        '''
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        '''

        # post-message-passing
        #Final layers used for classification
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, 50),
                                        nn.ReLU(),
                                        nn.Linear(50, output_dim))

        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        #Change this if you edit your structure
        self.num_layers = 1



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #Default if nodes does not have features
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        #Apply Convolutions

        for i in range(self.num_layers):
            x = self.net[i](x, edge_index)
            emb = x
            x = F.relu(x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        #post conv
        x = self.post_mp(x)

        #x = self.post_mp(x)


        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



def train(dataset, task, writer):

    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=256, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(100000):
        total_loss = 0
        model.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f} %".format(
                epoch, total_loss, test_acc*100))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()

task = 'graph'

model = train(dataset, task, writer)
