#SYMMETRIC POSITIVE DEFINITE MATRIX
import numpy as np
import torch
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys


#Compute Adjacency Matrix given Laplacian A
def computeAdjancency(A):
  A = -A;
  ind = np.diag_indices(A.shape[0])
  A[ind[0], ind[1]] = 0;
  return A


#Compute Laplacian of Adjacency Matrix A
def computeLaplacian(A):
    A = - A;
    sum_val = abs(torch.sum(A, dim=1))
    non_null_val = torch.where(sum_val > 0)[0];
    ind = np.diag_indices(A.shape[0])
    A[ind[0], ind[1]] = sum_val;
    A = A[non_null_val,:];
    A = A[:,non_null_val];
    A = A.type(torch.DoubleTensor)
    return A.cuda(), non_null_val;

def computeStrongConnections(A, param):
    #get the maximum of rows
    N = A.size(1)
    EXCLUDE_DIAG = (torch.eye(N)<1).cuda();
    A_noDiag = A*EXCLUDE_DIAG

    MaxAIJ = torch.max(torch.abs(A_noDiag),dim=1).values

    MaxAIJ_T = torch.t(MaxAIJ)
    MAX_MATRIX = MaxAIJ_T.repeat(N,1).t()

    #Compute which Elements (Diagonals Exluded) satifies the "Strong
    #Connection Requirement) for each node (row).
    S = torch.logical_and(torch.abs(A) >= (param*MAX_MATRIX), EXCLUDE_DIAG);
    B = torch.abs(A) >= (param*MAX_MATRIX);
    return S

def computeLambdas(F,U,S):

    N = S.size(1);
    #Compute mask for U nodes
    maskU = torch.zeros(N,N).cuda();
    #Get indices of Nodes in U

    maskU[torch.where(U==1)[0],:] = 1;
    #Create Mask for F Nodes
    maskF = torch.zeros(N,N).cuda();

    maskU[F,:] = 1;
    #Compute intersection
    SintU = (S*maskU).t();
    SintF = (S*maskF).t();

    #Compute lambdas according to formula
    lambdas = torch.sum(SintU,1) + 2*torch.sum(SintF,1);

    #Get only those elements belonging to U, and return lambdas;
    return lambdas*U.t()


def computePrologator(A, SUBNODES_PG):
    dim = A.size(1)
    SUBNODES = int(dim*SUBNODES_PG);
    S = computeStrongConnections(A, 0.25);
    F = np.array([], dtype='int');
    F = torch.from_numpy(F).cuda()
    C = np.array([], dtype='int');
    C = torch.from_numpy(C).cuda()
    U = torch.ones(dim,1).cuda();

    #neighbour matrix
    N = (A != 0).cuda();
    N = N * (torch.eye(dim)<1).cuda();
    #Compute lambdas according formula
    lambdas = computeLambdas(F,U,S);

    #SUBSPACE COARSENING
    for iteration in range(0,SUBNODES):
        index = torch.where(lambdas == torch.max(lambdas))[1][0];
        if(iteration ==0):
          C = index.unsqueeze(0);
        else:
          C = torch.cat((C,index.unsqueeze(0)))
        U[C] = 0;
        concatenate1 = torch.where(S[index,:]==1)[0]
        if(iteration == 0):
          F = concatenate1;
        else:
          F = torch.unique(torch.cat((F.unsqueeze(0),concatenate1.unsqueeze(0)),1)).cuda()
        U[F] = 0;
        lambdas = computeLambdas(F,U,S);

    #Compute the Prolongation Operator
    Prol = torch.zeros(dim,dim);

    #Cnodes Mask
    CNodes = torch.zeros(dim,dim).cuda();
    CNodes[C,:] = 1;
    CNodes[:,C]= 1;

    #Compute P (as in paper) as and AND element wise between CNodes and
    #Neighbours Nodes
    P = torch.logical_and(N,CNodes).cuda();

    dotProduct1 = A*N;
    secondProduct2 = A*P;

    #Compute weights of Prologation Operator for Fine nodes
    for i in F:
        Js = torch.where(P[i,:]==1)[0];

        for j in Js:
            firstDot = torch.sum(dotProduct1[i,:]);
            secondDot = torch.sum(secondProduct2[i,:]);
            Prol[i,j] = - (firstDot/secondDot)*(A[i,j]/A[i,i]);

    #Compute weights of Prologation Operator for Coarse nodes
    for i in C:
        Prol[i,i] = A[i,i];

    #Get only the columns that contain the "COARSEN NODES";

    Prol = Prol[:,C];

    Prol = Prol.type(torch.DoubleTensor)
    return Prol
