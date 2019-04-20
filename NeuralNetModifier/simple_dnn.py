import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, layer_size=(784, 500, 500, 10)):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(in_features=layer_size[0], out_features=layer_size[1], bias=True)
        self.dense2 = nn.Linear(in_features=layer_size[1], out_features=layer_size[2], bias=True)
        self.out_layer = nn.Linear(layer_size[2], layer_size[3])

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.sigmoid(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        x = self.out_layer(x)
        return F.log_softmax(x)


def train(epoch):

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    batch_size = 64
    model = Net()
    model.load_state_dict(torch.load("./model/simple_dnn_dict.pkl"))
    model.eval()
    # weight0 = model.state_dict()
    # weight1 = torch.Tensor(weight0)
    # print(weight0, type(weight0))
    # collections.OrderedDict()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 50):
        train(epoch)
        torch.save(model.state_dict(), "./model/simple_dnn_dict.pkl")
        test()

