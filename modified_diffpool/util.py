import torch
import numpy as np


# ---- NetworkX compatibility
def node_iter(G):
    return G.nodes

def node_dict(G):
    return G.nodes
# ---------------------------

def projection0(assign):

    assign.detach()

    proj = torch.zeros(assign.size()[0], assign.size()[2], assign.size()[1])

    pi = torch.argmax(assign, dim=2)
    for i in range(assign.size()[0]):

        cluster = [[] for _ in range(assign.size()[2])]
        for j in range(assign.size()[1]):
            cluster[pi[i,j]].append(j)

        for j in range(assign.size()[2]):
            if len(cluster[j])!=0: 
                for k in cluster[j]:
                    proj[i,j,k] = 1/len(cluster[j])

    return proj.float()

def projection1(assign):

    assign.detach()

    proj = torch.zeros(assign.size()[0], assign.size()[2], assign.size()[1])
    gamma = torch.zeros(assign.size()[0], assign.size()[2], assign.size()[2])

    pi = torch.argmax(assign, dim=2)
    for i in range(assign.size()[0]):

        cluster = [[] for _ in range(assign.size()[2])]
        for j in range(assign.size()[1]):
            cluster[pi[i,j]].append(j)

        for j in range(assign.size()[2]):
            if len(cluster[j])!=0: 
                for k in cluster[j]:
                    gamma[i,j,j] = 1/len(cluster[j])**2
                    proj[i,j,k] = 1/len(cluster[j])

        proj[i] = torch.linalg.pinv(proj[i]).t()

    proj = gamma@proj

    return proj, gamma

def comb_laplacian_np(adj):
    return np.diag(np.sum(adj,axis=0))-adj

def comb_laplacian_torch(adj):
    return torch.diag(torch.sum(adj,dim=0))-adj

def batch_comb_laplacian_torch(adj):
    laplacian = torch.zeros(adj.size())
    for i in range(adj.size()[0]):
        laplacian[i] = comb_laplacian_torch(adj[i])
    return laplacian

def QF(A,x):
    return (x.t()@A@x)

def RQ(A,x):
    return (x.t()@A@x)/(torch.abs(x.t()@x+1e-5))