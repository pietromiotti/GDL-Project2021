import argparse
import os
import json
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sklearn.metrics as metrics
import time
import cross_val
import encoders
import load_data
import util


def evaluate(dataset, model):
    
    model.eval()

    labels = []
    preds = []

    for _, data in enumerate(dataset):

        adj = Variable(data['adj'].float(), requires_grad=False)
        h0 = Variable(data['feats'].float(), requires_grad=False)
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False)

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print("Validation accuracy:", result['acc'])
    return result

def train(i, dataset, model, args, val_dataset):

    model.goren_loss_type = args.goren_loss_type

    if args.temp:
        print('temp = True')
        temps = []
        for k in range(args.num_epochs):
            temps.append(min(1,1e-2+(1+math.cos( (k*math.pi)/(args.num_epochs-1) ))/2))

    if args.optim == 'sgd':
        print('optimizer = sgd + cos annealing')
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.1, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4)
    else:
        print('optimizer = adam')
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)

    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}

    best_val_accs = []
    best_val_epochs = []
    val_accs = []
    val_results = []

    for epoch in range(args.num_epochs):
        print('Epoch: ', epoch)

        model.train()

        if args.temp:
            model.temperature = temps[epoch]

        total_time = 0
        avg_loss = 0.0

        for batch_idx, data in enumerate(dataset):

            begin_time = time.time()

            model.zero_grad()

            adj = Variable(data['adj'].float(), requires_grad=False)
            h0 = Variable(data['feats'].float(), requires_grad=False)
            laplacian = Variable(data['laplacian'].float(), requires_grad=False)
            eigenvectors = Variable(data['eigenvectors'].float(), requires_grad=False)
            label = Variable(data['label'].long())
            batch_num_nodes = data['num_nodes'].int().numpy()
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False)

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)

            loss = model.loss(ypred, label, adj, laplacian, eigenvectors, batch_num_nodes)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            avg_loss += loss

            elapsed = time.time() - begin_time
            total_time += elapsed

        if args.optim == 'sgd':
            scheduler.step()

        avg_loss /= batch_idx + 1

        #print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        #result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        #train_accs.append(result['acc'])
        #train_epochs.append(epoch)

        val_result = evaluate(val_dataset, model)
        val_results.append(val_result)
        val_accs.append(val_result['acc'])

        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])

    save = json.dumps(val_results)
    if args.goren_loss_type in [0,1]:
        f = open(os.path.dirname(os.path.abspath(__file__))+"/valres{}{}.json".format(args.goren_loss_type,i),"w")
    else:
        f = open(os.path.dirname(os.path.abspath(__file__))+"/valres{}.json".format(i),"w")
    f.write(save)
    f.close()

def benchmark_task_val(args):

    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    print('Using node labels')
    for G in graphs:
        for u in G.nodes():
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])

    for i in range(10):

        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)

        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim)

        train(i, train_dataset, model, args, val_dataset)

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')
    
    parser.add_argument('--goren_loss_type', type=int, default=0)
    parser.add_argument('--optim', type=str, default=None)
    parser.add_argument('--temp', type=bool, default=True)
    parser.add_argument('--neigen', type=int, default=10)


    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=130,
                        cuda='0',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                       )
    return parser.parse_args()

def main():
    benchmark_task_val(arg_parse())

if __name__ == "__main__":
    main()