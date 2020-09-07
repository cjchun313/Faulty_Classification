from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from data_loader import ImagevDatasetForClassi
from torch.utils.data import DataLoader

from models.resnet import ResNet18
from utils import monte_carlo_dropout, compute_acc, compute_confusion_matrix, compute_confidnce_interval

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

MODEL_PATH = '../pth/20200906/model-21-0.000000-97.7778.pth'

MEAN_VALUE = 0.006096079
VAR_VALUE = 0.05440523
STD_95_VALUE = 0.004255909
STD_99_VALUE = 0.006101583



def train(model, train_loader, optimizer):
    model.train()
    correct = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        data, target = data.to(DEVICE, dtype=torch.float), target.to(DEVICE)

        output = model(data)
        loss = criterion(output, target)

        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Batch Index: {}\tLoss: {:.6f}'.format(batch_idx, loss.item()))

    #print('Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

    train_loss = loss.item() / len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    return train_loss, train_acc



def evaluate(model, test_loader):
    model.eval()
    model = monte_carlo_dropout(model)

    correct = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE, dtype=torch.float), target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss = loss.item() / len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    return test_loss, test_acc



def predict(model, test_loader):
    model.eval()
    model = monte_carlo_dropout(model)

    total_acc, total_cm = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(DEVICE, dtype=torch.float), target.to(DEVICE)

            total_pred = []
            for i in range(128):
                output = model(data)
                prediction = output.max(1, keepdim=True)[1]
                total_pred.append(prediction.cpu().detach().numpy())

            total_pred = np.squeeze(np.array(total_pred))
            target = target.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            #print(np.mean(np.std(y_pred, axis=0)), np.std(np.std(y_pred, axis=0)))

            prediction = compute_confidnce_interval(np.std(total_pred, axis=0), prediction, MEAN_VALUE, STD_95_VALUE)

            acc = compute_acc(target, prediction)
            total_acc.append(acc)
            #print(acc)

            cm = compute_confusion_matrix(target, prediction)
            total_cm.append(cm)
            #print(cm)

    total_acc = np.mean(np.array(total_acc))
    total_cm = np.sum(np.array(total_cm), axis=0)

    return total_acc, total_cm


def save_model(modelpath, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, modelpath)

    print('model saved')



def load_model(modelpath, model, optimizer=None, scheduler=None):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])

    print('model loaded!')



def main(args):
    # train
    if args.mode == 'train':
        train_dataset = ImagevDatasetForClassi(mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

        val_dataset = ImagevDatasetForClassi(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(train_dataloader, val_dataloader)

        model = ResNet18()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # set optimizer
        optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

        acc_prev = 0.0
        for epoch in range(args.epoch):
            # train set
            loss, acc = train(model, train_dataloader, optimizer)
            # validate set
            val_loss, val_acc = evaluate(model, val_dataloader)

            print('Epoch:{}\tTrain Loss:{:.6f}\tTrain Acc:{:2.4f}'.format(epoch, loss, acc))
            print('Val Loss:{:.6f}\tVal Acc:{:2.4f}'.format(val_loss, val_acc))

            if val_acc > acc_prev:
                acc_prev = val_acc
                modelpath = '../pth/20200906/model-{:d}-{:.6f}-{:2.4f}.pth'.format(epoch, val_loss, val_acc)
                save_model(modelpath, model, optimizer, scheduler)

            # scheduler update
            scheduler.step()
    # evaluate
    elif args.mode == 'evaluate':
        val_dataset = ImagevDatasetForClassi(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(val_dataloader)

        model = ResNet18()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        #modelpath = '../pth/20200906/model.pth'
        load_model(MODEL_PATH, model)

        test_loss, test_acc = evaluate(model, val_dataloader)
        print('Test Loss:{:.6f}\tTest Acc:{:2.4f}'.format(test_loss, test_acc))
    # predict
    elif args.mode == 'predict':
        val_dataset = ImagevDatasetForClassi(mode='val2')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(val_dataloader)

        model = ResNet18()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # modelpath = '../pth/20200906/model.pth'
        load_model(MODEL_PATH, model)

        acc, cm = predict(model, val_dataloader)
        print('Accuracy:{:2.4f}'.format(acc))
        print(cm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of mini-batch samples',
        default=512,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--shuffle',
        help='True or False',
        default=True,
        type=bool)
    parser.add_argument(
        '--mode',
        help='train, evaluate, or predict',
        default='train',
        type=str)

    args = parser.parse_args()
    print(args)

    main(args)
