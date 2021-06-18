import argparse
import json
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--goren_loss_type', type=int, default=None)
    parser.add_argument('--path', type=str)
    return parser.parse_args()

def main():

    args = arg_parse()

    best_accs = []
    all_vals = []

    for j in range(10):

        accs =[]

        if args.goren_loss_type in [0,1]:
            with open(args.path+'valres{}{}.json'.format(args.goren_loss_type,j), "r") as fp:
                b = json.load(fp)

        else:
            with open(args.path+'valres{}.json'.format(j), "r") as fp:
                b = json.load(fp)

        best_acc = 0
        for k in b:
            accs.append(k['acc'])
            if k['acc']>best_acc:
                best_acc = k['acc']

        best_accs.append(best_acc)
        all_vals.append(np.array(accs))

    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)

    print('Accuracy: {}'.format(np.max(all_vals)))
    print('')
    print('Best accs over each fold:')
    print('max: {}'.format(np.max(best_accs)))
    print('min: {}'.format(np.min(best_accs)))
    print('mean: {}'.format(np.mean(best_accs)))
    print('std: {}'.format(np.std(best_accs)))

if __name__ == "__main__":
    main()