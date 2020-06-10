# example adapt from https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def predict(self, x):
        x = x.unsqueeze(0)
        net_param = next(self.parameters())
        x = x.to(net_param)

        y = self.forward(x)
        return y.argmax(dim=1).squeeze(0)  # get the index of the max log-probability

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class MyMNIST(MNIST):
    def evaluate(self, predictions, num_processes=1):
        TP = 0
        for i, sample in enumerate(self):
            target = sample[1]
            pred = predictions[str(i)]
            if target == pred:
                TP += 1
        return {'score': TP / len(self)}


if __name__ == '__main__':
    from torchvision import transforms
    from trainer.data import Subset
    from trainer.train import ArgumentParser, Trainer

    parser = ArgumentParser()
    args = parser.parse_args()

    net = Net()

    mnist_root = 'data/mnist'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets = dict(train=MyMNIST(mnist_root, True, download=True, transform=transform),
                    valid=MyMNIST(mnist_root, False, download=True, transform=transform),
                    test=MyMNIST(mnist_root, False, download=True, transform=transform)
                    )
    datasets = {k: Subset(d, slice(0, len(d), 10)) for k, d in datasets.items()}
    criterion = F.nll_loss
    trainer = Trainer(net, datasets, criterion, args)
    trainer.run()