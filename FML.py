from __future__ import print_function
from collections import namedtuple
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy import sparse
import scipy
import argparse
import glob
import os
import shutil
import time
import math
import pandas as pd
import numpy as np
import cPickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.dataset as dataset

parser = argparse.ArgumentParser(description='PyTorch SparseNN Training')
parser.add_argument('--gamma', default=0.99, type=float, metavar='G',
                    help='discount factor')
parser.add_argument('--seed', default=543, type=int, metavar='N',
                    help='random seed')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', default=1, type=int, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU for training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=1e-2, type=float, metavar='N',
                    help='initial learning rate')
parser.add_argument('--wd', default=1e-4, type=float, metavar='N',
                    help='weight decay')
parser.add_argument('--resume', default=0, type=int, metavar='N',
                    help='version of the latest checkpoint')
parser.add_argument('--steps', default=10, type=int, metavar='N',
                    help='number of gradient steps')
args = parser.parse_args()

if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc0 = nn.Linear(1, 40)
        self.fc1 = nn.Linear(40, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    global version
    version = args.resume or np.random.randint(1e9)
    print('=> version', version)

    meta_model = Model()
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.lr)
    pre_model = Model()
    pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.lr)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()

    if args.gpu:
        meta_model.cuda()
        pre_model.cuda()
        criterion.cuda()

    if args.resume:
        filename='../model/checkpoint_%s.pth.tar' % version
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            args.start_epoch = checkpoint['epoch']
            best_auc = checkpoint['best_auc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    train_loader = torch.utils.data.DataLoader(
        SyntheticDataset('train'), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        SyntheticDataset('test'), batch_size=args.batch_size, shuffle=False)

    best_loss = 100
    meta_res = []
    pre_res = []
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        pre_train(train_loader, pre_model, pre_optimizer, criterion, epoch)

        # evaluate on validation set
        loss = test(test_loader, pre_model, criterion, 'pre')
        pre_res.append(loss)
        pickle.dump(pre_res, open("../result/res_pt_sin.pickle", 
            'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # train for one epoch
        meta_train(train_loader, meta_model, meta_optimizer, criterion, epoch)

        # evaluate on validation set
        loss = test(test_loader, meta_model, criterion, 'meta')
        meta_res.append(loss)
        pickle.dump(meta_res, open("../result/res_ml_sin.pickle", 
            'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # remember best auc and save checkpoint
        is_best = loss[-1] < best_loss
        best_loss = min(loss[-1], best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': meta_model.state_dict(),
            'best_auc': best_loss,
            'optimizer': meta_optimizer.state_dict(),
        }, is_best)


def meta_train(train_loader, meta_model, meta_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    meta_model.train()

    end = time.time()
    for i, (inputs, outputs) in enumerate(train_loader):
        n = len(outputs)
        for k in range(n):
            model = Model()
            model.load_state_dict(meta_model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            m = len(outputs[k])
            m_train = m / 2
            x, y = inputs[k][:m_train,:], outputs[k][:m_train]
            x_train_var = torch.autograd.Variable(x, requires_grad=False)
            y_train_var = torch.autograd.Variable(y)
            x, y = inputs[k][m_train:,:], outputs[k][m_train:]
            x_test_var = torch.autograd.Variable(x, requires_grad=False)
            y_test_var = torch.autograd.Variable(y)
            
            if args.gpu:
                model.cuda()
                x_train_var = x_train_var.cuda()
                y_train_var = y_train_var.cuda()
                x_test_var = x_test_var.cuda()
                y_test_var = y_test_var.cuda()

            # compute output
            output = model(x_train_var)
            loss = criterion(output, y_train_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute output
            output = model(x_test_var)
            loss = criterion(output, y_test_var)
            losses.update(loss.data.cpu()[0], y.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for metap, p in zip(meta_model.parameters(), model.parameters()):
                if k == 0:
                    metap.grad = p.grad
                else:
                    metap.grad.data += p.grad.data
        meta_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.log_interval == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #            epoch, i, len(train_loader), batch_time=batch_time, 
        #            loss=losses))


def pre_train(train_loader, pre_model, pre_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    pre_model.train()

    end = time.time()
    for i, (inputs, outputs) in enumerate(train_loader):
        n = len(outputs)
        pre_optimizer.zero_grad()
        for k in range(n):
            x, y = inputs[k], outputs[k]
            x_var = torch.autograd.Variable(x, requires_grad=False)
            y_var = torch.autograd.Variable(y)
            
            if args.gpu:
                x_var = x_var.cuda()
                y_var = y_var.cuda()

            # compute output
            output = pre_model(x_var)
            loss = criterion(output, y_var)
            losses.update(loss.data.cpu()[0], y.size(0))

            # compute gradient and do SGD step
            loss.backward()
        pre_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.log_interval == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #            epoch, i, len(train_loader), batch_time=batch_time, 
        #            loss=losses))


def test(test_loader, base_model, criterion, tag):
    batch_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.steps)]

    # switch to evaluate mode
    base_model.eval()

    end = time.time()
    for i, (inputs, outputs) in enumerate(test_loader):
        n = len(outputs)
        for k in range(n):
            model = Model()
            model.load_state_dict(base_model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            m = len(outputs[k])
            m_train = m / 2
            x, y = inputs[k][:m_train,:], outputs[k][:m_train]
            x_train_var = torch.autograd.Variable(x, requires_grad=False)
            y_train_var = torch.autograd.Variable(y)
            x, y = inputs[k][m_train:,:], outputs[k][m_train:]
            x_test_var = torch.autograd.Variable(x, requires_grad=False)
            y_test_var = torch.autograd.Variable(y)
            
            if args.gpu:
                model.cuda()
                x_train_var = x_train_var.cuda()
                y_train_var = y_train_var.cuda()
                x_test_var = x_test_var.cuda()
                y_test_var = y_test_var.cuda()

            for step in range(args.steps):
                # compute output
                output = model(x_test_var)
                loss = criterion(output, y_test_var)
                losses[step].update(loss.data.cpu()[0], y.size(0))

                if step + 1 == args.steps:
                    break

                # compute output
                output = model(x_train_var)
                loss = criterion(output, y_train_var)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(test_loader), batch_time=batch_time, 
                   loss=losses[-1]), tag)

    return [loss.avg for loss in losses]


class SyntheticDataset(dataset.Dataset):

    def __init__(self, filename):
        path = "../data/%s.pickle" % filename
        if not os.path.isfile(path) and filename == 'train':
            n, m = 1000, 20
            n_train = int(n * 0.8)
            amp = np.random.uniform(0.1, 5.0, [n]).astype(np.float32)
            phase = np.random.uniform(0, np.pi, [n]).astype(np.float32)
            inputs = np.zeros([n, m, 1]).astype(np.float32)
            outputs = np.zeros([n, m, 1]).astype(np.float32)
            for func in range(n):
                inputs[func] = np.random.uniform(-5.0, 5.0, [m, 1])
                outputs[func] = amp[func] * np.sin(inputs[func] - phase[func])
            pickle.dump((inputs[:n_train], outputs[:n_train]), 
                open("../data/train.pickle", 'wb'), 
                protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((inputs[n_train:], outputs[n_train:]), 
                open("../data/test.pickle", 'wb'), 
                protocol=pickle.HIGHEST_PROTOCOL)

        self.inputs, self.outputs = pickle.load(open(path, "rb"))

    def __getitem__(self, index):
        inputs = self.inputs[index]
        outputs = self.outputs[index]

        return (inputs, outputs)

    def __len__(self):
        return len(self.outputs)


def save_checkpoint(state, is_best):
    filename = '../model/checkpoint_%s.pth.tar' % version
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../model/best_%s.pth.tar' % version)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()