# encoding=utf-8
"""
    Created on 21:11 2018/11/8 
    @author: Jindong Wang
"""


from itertools import accumulate
import data_preprocess
import matplotlib.pyplot as plt
import network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--data_folder', type=str, default='../')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    return args


def train(model, optimizer, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()

    for e in range(args.nepoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, target in train_loader:
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)
            output = model(sample)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)

        # Testing
        acc_test = valid(model, test_loader)
        print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {float(correct) * 100 / total:.2f}')
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        np.savetxt('result.csv', result_np, fmt='%.2f', delimiter=',')


def valid(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig('plot.png')


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = data_preprocess.load(
        args.data_folder, batch_size=args.batchsize)
    model = net.Network().to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(
    ), lr=args.lr, momentum=args.momentum)
    train(model, optimizer, train_loader, test_loader)
    result = np.array(result, dtype=float)
    np.savetxt('result.csv', result, fmt='%.2f', delimiter=',')
    plot()
