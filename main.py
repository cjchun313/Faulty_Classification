import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import argparse

from data_loader import data_loader
from models.resnet import ResNet18


def train(model, train_loader, optimizer, DEVICE):
    model.train()

    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print('Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

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
    if args.train is True:
        train_loader = data_loader(train=True, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
        val_loader = data_loader(train=False, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
    else:
        test_loader = data_loader(train=args.train, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    print('device:', DEVICE)

    if args.train is True:
        model = ResNet18().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        acc_prev = 0
        for epoch in tqdm(range(args.epoch)):
            train(model, train_loader, optimizer, DEVICE)
            loss, acc = evaluate(model, val_loader, DEVICE)
            print('epoch:{}\tval loss:{:.6f}\tval acc:{:2.4f}'.format(epoch, loss, acc))
            if acc > acc_prev:
                acc_prev = acc

                savepath = '../pth/resnet_model-{:d}-{:.4f}.pth'.format(epoch, acc)
                torch.save(model.state_dict(), savepath)
    else:
        model = ResNet18().to(DEVICE)
        model.load_state_dict(torch.load('../pth/resnet_model-59-98.5250.pth'))

        loss, acc = evaluate(model, test_loader, DEVICE)
        print('test loss:{:.6f}\ttest acc:{:2.4f}'.format(loss, acc))

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
        default=1,
        type=int)
    parser.add_argument(
        '--shuffle',
        help='if 1, shuffle',
        default=1,
        type=int)
    parser.add_argument(
        '--train',
        help='if 1, train, or evaluate',
        default=1,
        type=int)
    parser.add_argument(
        '--target_idx',
        help='target data index, dont care if train is true',
        default=0,
        type=int)

    args = parser.parse_args()
    print(args)

    main(args)
