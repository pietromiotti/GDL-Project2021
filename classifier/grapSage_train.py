import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from graphSage import SageConv

dataset = Planetoid(root='/tmp/Planetoid', name='Cora')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
use_cuda_if_available = False

class Net(torch.nn.Module):
    def __init__(self, num_feat, num_class):
        super(Net, self).__init__()

        self.conv1 = SageConv(num_feat,
                             num_class)


    def forward(self, x, index):
        out = self.conv1(self.x, self.index)
        return F.log_softmax(out, dim=1)


def main():
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    model, data = Net(dataset.num_features, dataset.num_classes).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = test_acc = 0
    for epoch in range(1,100):
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

        model.eval()
        logits, accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        _, val_acc, tmp_test_acc = accs
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'

        if epoch % 10 == 0:
            print(log.format(epoch, best_val_acc, test_acc))



if __name__ == '__main__':

    main()
