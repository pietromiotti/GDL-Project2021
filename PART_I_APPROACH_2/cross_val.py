import networkx as nx
import numpy as np
import torch

import pickle
import random
from easyAMG import computeLaplacian, computePrologator,computeAdjancency

from graph_sampler import GraphSampler

def prepare_val_data(graphs, args, val_idx, max_nodes=0):
    random.shuffle(graphs)
    val_size = len(graphs) // 10

    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]

    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))


    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    print("performing restrictions...")

    #Precomputation of Graphs.
    #We could only get 460 samples for training because the RAM of Colab does not permit us to do otherwise
    dataset_sampler_new = []
    i=0
    maxDATA = 460;
    for data in dataset_sampler:

        #For each graph, compute the Prolongation Operator, the new Adjacency Matrix and the new Feature Matrix
        temp = {}
        temp['adj2'] = data['adj']
        temp['feats2'] = data['feats']
        temp['assign_feats2'] = data['assign_feats']
        temp['num_nodes2'] = data['num_nodes']

        #Compute Laplacian
        A, real_index = computeLaplacian(torch.from_numpy(data['adj']))
        real_features = torch.from_numpy(data['feats'][real_index,:])
        real_assign_feats = torch.from_numpy(data['assign_feats'][real_index,:])

        real_features = real_features.cuda()
        real_assign_feats = real_assign_feats.cuda()

        #Prolongatiom
        P = (computePrologator(A,1)).cuda();

        #Coarsen A
        A = torch.matmul(P.t(),torch.matmul(A,P));
        A = computeAdjancency(A);
        A = A.cpu()

        #Coarsen X
        x_h = torch.matmul(P.t(), real_features);
        x_h = x_h.cpu()

        #Coarsen x_ass
        x_ass_h = torch.matmul(P.t(), real_assign_feats);
        x_ass_h = x_ass_h.cpu()


        #Compute Padding
        A_padded = np.zeros(shape=(data['adj'].shape[0],data['adj'].shape[1]));
        A_padded[:A.shape[0],:A.shape[1]] = A.numpy()

        x_h_padded = np.zeros(shape=(data['feats'].shape[0], data['feats'].shape[1]))
        x_h_padded[:x_h.shape[0],:x_h.shape[1]] = x_h.numpy()

        x_assigned_padded = np.zeros(shape=(data['assign_feats'].shape[0], data['assign_feats'].shape[1]))
        x_assigned_padded[:x_ass_h.shape[0], :x_ass_h.shape[1]] = x_ass_h.numpy()
        temp['num_nodes1'] = A.shape[0]
        temp['adj1'] = A_padded
        temp['feats1'] = x_h_padded
        temp['assign_feats1'] = x_assigned_padded
        temp['label'] = data['label']

        dataset_sampler_new.append(temp)
        if(i>maxDATA):
          break;
        else:
          i= i+1;


    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler_new,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)


    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)

    #Same operations as for the training samples
    dataset_sampler_val = []
    for data in dataset_sampler:
      temp = {}
      temp['adj2'] = data['adj']
      temp['feats2'] = data['feats']
      temp['assign_feats2'] = data['assign_feats']
      temp['num_nodes2'] = data['num_nodes']

      A, real_index = computeLaplacian(torch.from_numpy(data['adj']))
      real_features = torch.from_numpy(data['feats'][real_index,:])
      real_assign_feats = torch.from_numpy(data['assign_feats'][real_index,:])

      real_features = real_features.cuda()
      real_assign_feats = real_assign_feats.cuda()

      P = (computePrologator(A,1)).cuda();
      A = torch.matmul(P.t(),torch.matmul(A,P));
      A = computeAdjancency(A);
      A = A.cpu()
      x_h = torch.matmul(P.t(), real_features);
      x_h = x_h.cpu()
      x_ass_h = torch.matmul(P.t(), real_assign_feats);
      x_ass_h = x_ass_h.cpu()

      A_padded = np.zeros(shape=(data['adj'].shape[0],data['adj'].shape[1]));
      A_padded[:A.shape[0],:A.shape[1]] = A.numpy()

      x_h_padded = np.zeros(shape=(data['feats'].shape[0], data['feats'].shape[1]))
      x_h_padded[:x_h.shape[0],:x_h.shape[1]] = x_h.numpy()

      x_assigned_padded = np.zeros(shape=(data['assign_feats'].shape[0], data['assign_feats'].shape[1]))
      x_assigned_padded[:x_ass_h.shape[0], :x_ass_h.shape[1]] = x_ass_h.numpy()
      temp['num_nodes1'] = A.shape[0]
      temp['adj1'] = A_padded
      temp['feats1'] = x_h_padded
      temp['assign_feats1'] = x_assigned_padded
      temp['label'] = data['label']
      dataset_sampler_val.append(temp)

    print("Restrictions Computed")
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
