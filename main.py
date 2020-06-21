import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import argparse

from data_loader import data_loader
from models.resnet import ResNet18
from freeze import freeze_model

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
    if args.train == 1:
        if args.target_size == 1:
            train_loader = data_loader(size=1, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
            val_loader = data_loader(size=0, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
        else:
            train_loader = data_loader(size=0, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
            val_loader = data_loader(size=1, target_idx=0, shuffle=args.shuffle, batch_size=args.batch_size)
    elif args.train == 2:
        if args.target_size == 1:
            train_loader = data_loader(size=1, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)
            val_loader = data_loader(size=0, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)
        else:
            train_loader = data_loader(size=0, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)
            val_loader = data_loader(size=1, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)
    else:
        if args.target_size == 1:
            test_loader = data_loader(size=1, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)
        else:
            test_loader = data_loader(size=0, target_idx=args.target_idx, shuffle=args.shuffle, batch_size=args.batch_size)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    print('device:', DEVICE)

    if args.train == 1:
        # source train
        model = ResNet18().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        acc_prev = 0
        for epoch in tqdm(range(args.epoch)):
            train(model, train_loader, optimizer, DEVICE)
            loss, acc = evaluate(model, val_loader, DEVICE)
            print('epoch:{}\tval loss:{:.6f}\tval acc:{:2.4f}'.format(epoch, loss, acc))
            if acc > acc_prev:
                acc_prev = acc

                savepath = '../pth/resnet_model_source.pth'
                torch.save(model.state_dict(), savepath)
    elif args.train == 2:
        # transfer learning
        model = ResNet18().to(DEVICE)
        model.load_state_dict(torch.load('../pth/resnet_model_source.pth', map_location=torch.device(DEVICE)))
        model = freeze_model(model, num_layers=6)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        acc_prev = 0
        for epoch in tqdm(range(args.epoch)):
            train(model, train_loader, optimizer, DEVICE)
            loss, acc = evaluate(model, val_loader, DEVICE)
            print('epoch:{}\tval loss:{:.6f}\tval acc:{:2.4f}'.format(epoch, loss, acc))
            if acc > acc_prev:
                acc_prev = acc

                savepath = '../pth/resnet_model_transfer_target' + str(args.target_idx) + '.pth'
                torch.save(model.state_dict(), savepath)
    else:
        # evaluate
        model = ResNet18().to(DEVICE)
        if args.model_idx == 0:
            model.load_state_dict(torch.load('../pth/resnet_model_source.pth', map_location=torch.device(DEVICE)))
        elif args.model_idx == 1:
            model.load_state_dict(torch.load('../pth/resnet_model_transfer_target1.pth', map_location=torch.device(DEVICE)))
        elif args.model_idx == 2:
            model.load_state_dict(torch.load('../pth/resnet_model_transfer_target2.pth', map_location=torch.device(DEVICE)))
        elif args.model_idx == 3:
            model.load_state_dict(torch.load('../pth/resnet_model_transfer_target3.pth', map_location=torch.device(DEVICE)))
        else:
            model.load_state_dict(torch.load('../pth/resnet_model_source.pth', map_location=torch.device(DEVICE)))

        loss, acc = evaluate(model, test_loader, DEVICE)
        print('test loss:{:.6f}\ttest acc:{:2.4f}'.format(loss, acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of mini-batch samples',
        default=64,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=1,
        type=int)
    parser.add_argument(
        '--shuffle',
        help='if 1, shuffle is true',
        default=1,
        type=int)
    parser.add_argument(
        '--train',
        help='evaluation : 0, train : 1, transfer : 2',
        default=2,
        type=int)
    parser.add_argument(
        '--target_size',
        help='1: big(16000), 0: small(4000)',
        default=1,
        type=int)
    parser.add_argument(
        '--target_idx',
        help='0: source, 1: target1, 2: target2, 3: target3',
        default=1,
        type=int)
    parser.add_argument(
        '--model_idx',
        help='0: source, 1: target1, 2: target2, 3: target3',
        default=0,
        type=int)

    args = parser.parse_args()
    print(args)

    main(args)
