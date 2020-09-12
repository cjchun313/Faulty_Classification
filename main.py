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
from utils import monte_carlo_dropout, compute_acc, compute_confusion_matrix, compute_confidnce_interval, compute_logit_distance
from utils import extract_only_one_class, compute_euclidean_distance, compute_gumbel_r_mom, compute_gumbel_pdf, compute_gumbel_cdf, update_logit_from_cdf

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



def predict(model, test_loader, seed):
    model.eval()
    #model = monte_carlo_dropout(model)

    cnt = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(DEVICE, dtype=torch.float), target.to(DEVICE)

            output = model(data)
            prediction = output.max(1, keepdim=True)[1]

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            #prediction = prediction.cpu().detach().numpy()

            if cnt == 0:
                logit_0 = extract_only_one_class(target, output, 0)
                logit_1 = extract_only_one_class(target, output, 1)
                logit_2 = extract_only_one_class(target, output, 2)
                logit_3 = extract_only_one_class(target, output, 3)
                logit_4 = extract_only_one_class(target, output, 4)

                cnt += 1
            else:
                logit_0 = np.concatenate((logit_0, extract_only_one_class(target, output, 0)), axis=0)
                logit_1 = np.concatenate((logit_1, extract_only_one_class(target, output, 1)), axis=0)
                logit_2 = np.concatenate((logit_2, extract_only_one_class(target, output, 2)), axis=0)
                logit_3 = np.concatenate((logit_3, extract_only_one_class(target, output, 3)), axis=0)
                logit_4 = np.concatenate((logit_4, extract_only_one_class(target, output, 4)), axis=0)

                cnt += 1

            #print(output)

            #acc = compute_acc(target, prediction)
            #print(acc)

            #cm = compute_confusion_matrix(target, prediction)
            #print(cm)

    logit_0 = np.squeeze(np.array(logit_0))
    logit_1 = np.squeeze(np.array(logit_1))
    logit_2 = np.squeeze(np.array(logit_2))
    logit_3 = np.squeeze(np.array(logit_3))
    logit_4 = np.squeeze(np.array(logit_4))
    #print(logit_0.shape, logit_1.shape, logit_2.shape, logit_3.shape, logit_4.shape)

    mean_logit_0 = np.mean(logit_0, axis=0)
    mean_logit_1 = np.mean(logit_1, axis=0)
    mean_logit_2 = np.mean(logit_2, axis=0)
    mean_logit_3 = np.mean(logit_3, axis=0)
    mean_logit_4 = np.mean(logit_4, axis=0)
    print(mean_logit_0)
    print(mean_logit_1)
    print(mean_logit_2)
    print(mean_logit_3)
    print(mean_logit_4)

    logit_0 = compute_euclidean_distance(logit_0, mean_logit_0)
    logit_1 = compute_euclidean_distance(logit_1, mean_logit_1)
    logit_2 = compute_euclidean_distance(logit_2, mean_logit_2)
    logit_3 = compute_euclidean_distance(logit_3, mean_logit_3)
    logit_4 = compute_euclidean_distance(logit_4, mean_logit_4)
    #print(logit_0.shape, logit_1.shape, logit_2.shape, logit_3.shape, logit_4.shape)

    loc_0, scale_0 = compute_gumbel_r_mom(logit_0)
    loc_1, scale_1 = compute_gumbel_r_mom(logit_1)
    loc_2, scale_2 = compute_gumbel_r_mom(logit_2)
    loc_3, scale_3 = compute_gumbel_r_mom(logit_3)
    loc_4, scale_4 = compute_gumbel_r_mom(logit_4)
    print(loc_0, scale_0)
    print(loc_1, scale_1)
    print(loc_2, scale_2)
    print(loc_3, scale_3)
    print(loc_4, scale_4)

    #x = np.arange(0, 10, 0.01)
    #print(x.shape)
    #print(compute_gumbel_cdf(x, loc=loc_0, scale=scale_0))



def predict2(model, test_loader, seed):
    model.eval()
    #model = monte_carlo_dropout(model)

    mean_logits = [
        [5.1635995,     -6.4268346,     2.3972373,      -1.5858468,     -3.5654037],
        [-2.2015066,    37.41395,       -5.1897492,     -35.593792,     11.070286],
        [-3.6411288,    -11.667751,     11.988806,      0.8579022,      -6.9625316],
        [-4.024851,     -8.402394,      -3.8406122,     8.253992,       -3.2618928],
        [-3.8793495,    -0.35262492,    -6.2018995,     -3.8200183,     10.935963]
    ]
    mean_logits = np.array(mean_logits)
    #print(mean_logits.shape)

    total_acc, total_cm = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(DEVICE, dtype=torch.float), target.to(DEVICE)

            output = model(data)
            prediction = output.max(1, keepdim=True)[1]

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()

            logit_0 = compute_euclidean_distance(output, mean_logits[0])
            logit_1 = compute_euclidean_distance(output, mean_logits[1])
            logit_2 = compute_euclidean_distance(output, mean_logits[2])
            logit_3 = compute_euclidean_distance(output, mean_logits[3])
            logit_4 = compute_euclidean_distance(output, mean_logits[4])
            #print(logit_0.shape, logit_1.shape, logit_2.shape, logit_3.shape, logit_4.shape)

            cdf_0 = compute_gumbel_cdf(logit_0, loc=1.23504581711539, scale=0.890690418041288)
            cdf_1 = compute_gumbel_cdf(logit_1, loc=4.69870809919228, scale=2.72121064947818)
            cdf_2 = compute_gumbel_cdf(logit_2, loc=4.23566709581643, scale=2.09320085466454)
            cdf_3 = compute_gumbel_cdf(logit_3, loc=1.52365110518514, scale=0.771770951799924)
            cdf_4 = compute_gumbel_cdf(logit_4, loc=3.93608164456129, scale=1.97732504677074)
            #print(cdf_0.shape, cdf_1.shape, cdf_2.shape, cdf_3.shape, cdf_4.shape)

            update_output = update_logit_from_cdf(output, cdf_0, cdf_1, cdf_2, cdf_3, cdf_4)
            #print(update_output.shape)

            update_pred = np.argmax(update_output, axis=-1)
            #print(update_pred.shape)

            acc = compute_acc(target, update_pred)
            total_acc.append(acc)
            #print(acc)

            cm = compute_confusion_matrix(target, update_pred)
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

        acc, cm = predict2(model, val_dataloader, seed=0)
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
        default='predict',
        type=str)
    parser.add_argument(
        '--seed',
        help='seed number',
        default=0,
        type=int)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)
