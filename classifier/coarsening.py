import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import torch_geometric.utils as pyg_utils
from scipy import sparse
from sklearn import metrics
import pyamg
import torch
import numpy as np
from pyamg.gallery import poisson
from pyamg.multilevel import multilevel_solver
from pyamg.strength import classical_strength_of_connection
from pyamg.classical import direct_interpolation
from pyamg.classical.split import RS
from pyamg.aggregation.aggregate import lloyd_aggregation


#COARSENING FUNCTION BASED ON AMG

def coarseningGraph(x, edge_index):
    #x, edge_index, batch = batch.x, batch.edge_index, batch.batch
    L = pyg_utils.to_scipy_sparse_matrix(edge_index)

    
    Lcsr = sparse.csr_matrix(L)
    Xcsr =  sparse.csr_matrix(x.numpy())

    #here you can specify the max number of coarsening grids and the min number of nodes that a coarse garph can have
    ml = pyamg.smoothed_aggregation_solver(Lcsr, max_levels=2, max_coarse=1, keep=True)

    # AggOp[i,j] is 1 iff node i belongs to aggregate j
    AggOp = ml.levels[0].AggOp

    #Projection of Features
    xH = (AggOp.T).dot(Xcsr)

    #Projection of Laplacian R * L * P
    LH = (AggOp.T).dot(Lcsr.dot(AggOp))


    newLaplace = pyg_utils.from_scipy_sparse_matrix(LH)
    newX = pyg_utils.from_scipy_sparse_matrix(xH)

    return newX, newLaplace
