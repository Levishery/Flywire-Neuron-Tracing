import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import random
from torch import nn
import argparse
import sys
import os
import time

import numpy as np


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden1 layer
        self.hidden2 = torch.nn.Linear(n_hidden[0], n_hidden[1])  # hidden2 layer
        self.predict = torch.nn.Linear(n_hidden[1], n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)  # linear output
        return x

def getArg():
    parser = argparse.ArgumentParser(description='Conv model')
    parser.add_argument('--data_path', default='',
                        help='')
    parser.add_argument('--model', default='CNN',
                        help='')
    parser.add_argument('--EPOCH', default=400, type=int,
                        help='EPOCH to train')
    parser.add_argument('--N', default=5, type=int,
                        help='System order')
    parser.add_argument('--test_only', action='store_true',
                        help='output test result only')
    parser.add_argument('--pretrain', default='.',
                        help='pretrained model name')
    parser.add_argument('--name', default='Conv',
                        help='file name to save the model')

    args = parser.parse_args()
    return args

def loadData(test_only, data_path, pretrain, name):

    print('Preparing data...')
    data_list = torch.load(data_path)

    if test_only:
        statistics = torch.load('../Experiment/' + pretrain + '/statistics.pth')
        mean = statistics['mean']
        std = statistics['std']
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = []
        data_test = data_list[0:len(data_list)]
    else:
        mean = np.asarray([d.y[0][0] for d in data_list]).mean()
        std = np.asarray([d.y[0][0] for d in data_list]).std()
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = data_list
        data_test = []
        torch.save({'mean': mean, 'std': std}, '../Experiment/' + name + '/statistics.pth')

    return data_train, data_test, [std, mean]

def main():

    print(sys.argv)
    args = getArg()
    global prediction
    if not os.path.exists('../Experiment/' + args.name):
        os.makedirs('../Experiment/' + args.name)
    N = args.N
    data_train, data_test, [std, mean] = loadData(args.test_only, args.data_path, args.pretrain, args.name)
    if args.test_only:
        data_ = data_test
    else:
        data_ = data_train

    data_w = np.zeros([len(data_), N])
    data_a_upper = np.zeros([len(data_), int((N-1)/2*N)])
    data_a_lower = np.zeros([len(data_), int((N-1)/2*N)])
    data_MOCU = np.zeros([len(data_), 1])
    for i in range(len(data_)):
        data_w[i, :] = np.array(data_[i].x.squeeze())
        data_a_upper[i, :] = matrix2value(np.array(EdgeAtt2matrix(data_[i].edge_attr[:, 1], N), dtype=np.float64))
        data_a_lower[i, :] = matrix2value(np.array(EdgeAtt2matrix(data_[i].edge_attr[:, 0], N), dtype=np.float64))
        data_MOCU[i, 0] = data_[i].y
    data_x = np.concatenate((data_w, data_a_upper, data_a_lower), axis=1)
    model = Net(n_feature=N*N, n_hidden=[400, 200], n_output=1)

    data_y = data_MOCU
    if args.test_only:  # train_x/y will not be used
        train_x = data_x
        test_x = data_x
        train_y = data_y
        test_y = data_y
    else:
        train_x = data_x[0:int(len(data_)*0.96), :]
        test_x = data_x[int(len(data_)*0.96):len(data_), :]
        train_y = data_y[0:int(len(data_)*0.96), :]
        test_y = data_y[int(len(data_)*0.96):len(data_), :]

    # numpy to tensor
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    print('Making model')

    model.cuda()
    if args.pretrain != '.':
        model.load_state_dict(torch.load('../Experiment/' + args.pretrain + '/model.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 256
    EPOCH = args.EPOCH

    test_MSE = np.zeros(EPOCH)
    train_MSE = np.zeros(EPOCH)

    torch_dataset = Data.TensorDataset(train_x, train_y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, )

    test_x = test_x.cuda()
    test_y = test_y.cuda()
    # test_y = test_y.unsqueeze(1)

    # start training
    if args.test_only:
        EPOCH = 1
    for epoch in range(EPOCH):
        if not args.test_only:
            train_MSE_step = []
            for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
                # train
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                prediction = model(batch_x)  # [batch_size, 1]
                # batch_y = batch_y.unsqueeze(1)  # Without Unsqueeze, PyTorch will broadcast [batch_size] to a wrong
                # tensor in the MSE loss

                loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)

                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                train_MSE_step.append(loss.cpu())

            train_MSE[epoch] = sum(train_MSE_step) / len(train_MSE_step) * std * std
            print('epoch %d training MSE loss: %f' % (epoch, train_MSE[epoch]))

        # test
        start_time = time.time()
        prediction = model(test_x)
        test_time = time.time() - start_time
        loss = loss_func(prediction, test_y) * std * std
        print('epoch %d test MSE: %f, time: %f' % (epoch, loss, test_time))
        test_MSE[epoch] = loss
        if epoch > 5 and loss < min(test_MSE[0:epoch]):
            torch.save(model.state_dict(), '../Experiment/' + args.name + '/model.pth')

    exp_MSE = 0.0158*0.001
    var = std
    print(
        f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: {exp_MSE:.12f};   data variance: {var:.12f}")
    exp_MSE = np.full(len(train_MSE), exp_MSE)
    plt.plot(train_MSE[1:], 'r', label="train")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('../Experiment/' + args.name + '/train.png')
    print(train_MSE)
    print(test_MSE)
    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('../Experiment/' + args.name + '/train2.png')

    torch.save(model.state_dict(), '../Experiment/' + args.name + '/model.pth')


if __name__ == '__main__':
    main()
