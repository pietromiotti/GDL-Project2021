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

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)


    #PERFORMING THE RESTRICTIONS usign AMG
    print("performing restrictions...")
    dataset_sampler_new = []
    i=0
    maxDATA = 490;
    for data in dataset_sampler:
        #since it was not possible to edit directly data_sample object, we recreated new ones by adding also the Prolungation Operator
        temp = {}
        temp['adj'] = data['adj']
        temp['feats'] = data['feats']
        temp['assign_feats'] = data['assign_feats']
        temp['num_nodes'] = data['num_nodes']

        #Compute Laplacian
        A, real_index = computeLaplacian(torch.from_numpy(data['adj']))

        #Compute Prolungator
        P = (computePrologator(A,1)).cuda();
        A = torch.matmul(P.t(),torch.matmul(A,P));
        A = computeAdjancency(A);
        A = A.cpu()
        P = P.cpu()

        P_padded = np.zeros(shape=(data['adj'].shape[0],data['adj'].shape[1]));
        P_padded[:P.shape[0],:P.shape[1]] = P.numpy()

        temp['label'] = data['label']
        temp['prol'] = P_padded;

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

    dataset_sampler_val = []
    for data in dataset_sampler:
      temp = {}
      temp['adj'] = data['adj']
      temp['feats'] = data['feats']
      temp['assign_feats'] = data['assign_feats']
      temp['num_nodes'] = data['num_nodes']

      #compute Laplacian
      A, real_index = computeLaplacian(torch.from_numpy(data['adj']))

      #Compute Prolungator
      P = (computePrologator(A,1)).cuda();

      A = torch.matmul(P.t(),torch.matmul(A,P));
      A = computeAdjancency(A);
      A = A.cpu()
      P = P.cpu()

      #Padding Prolungator
      P_padded = np.zeros(shape=(data['adj'].shape[0],data['adj'].shape[1]));
      P_padded[:P.shape[0],:P.shape[1]] = P.numpy()

      temp['label'] = data['label']
      temp['prol'] = P_padded;
      dataset_sampler_val.append(temp)

    print("restriction Computed ")
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
