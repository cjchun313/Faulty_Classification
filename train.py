import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import argparse

from data_loader import data_loader
from models.resnet import ResNet18


def train(model, train_loader, optimizer, epoch, DEVICE):
    model.train()

    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))


def evaluate(model, test_loader, DEVICE):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction ='sum').item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def main(args):
    train_loader = data_loader(train=True, batch_size=args.batch_size)
    val_loader = data_loader(train=False)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    print('device:', DEVICE)

    model = ResNet18().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    acc_prev = 0
    for epoch in tqdm(range(args.epoch + 1)):
        train(model, train_loader, optimizer, epoch, DEVICE)
        loss, acc = evaluate(model, val_loader, DEVICE)
        if acc > acc_prev:
            acc_prev = acc

            savePath = './pth/resnet_model-{:4d}-{:.4f}.pth'.format(epoch, loss)
            torch.save(model.state_dict(), savePath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
    parser.add_argument(
        '--epoch',
        help='number of training iterations',
        default=1000,
        type=int)

    args = parser.parse_args()
    print(args)

    main(args)