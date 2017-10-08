#!/usr/bin/env python
from __future__ import print_function

from itertools import count
import glob
import math
import numpy as np
import os
import pandas as pd
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
import re

single_results_dir = './single_model_results/'
dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def conv3x3_1d(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):

    def __init__(self, block, layers, in_channels, num_classes=1):
        self.inplanes = 64
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18_1d(**kwargs):
    model = ResNet1D(BasicBlock1D, [2, 2, 2, 2], **kwargs)
    return model


def make_output_month(pred, train_y_mean, train_y_std, test_id):
    y_pred_10 = []
    y_pred_11 = []
    y_pred_12 = []
    for i in range(0, pred.shape[0], 3):
        y_pred_10.append(str(round(pred[i] * train_y_std + train_y_mean,4)))
        y_pred_11.append(str(round(pred[i+1] * train_y_std + train_y_mean,4)))
        y_pred_12.append(str(round(pred[i+2] * train_y_std + train_y_mean,4)))
    y_pred_10=np.array(y_pred_10)
    y_pred_11=np.array(y_pred_11)
    y_pred_12=np.array(y_pred_12)

    # print(pred.shape[0])
    # print(test_id.shape)
    # print(y_pred_10.shape)
    # print(y_pred_11.shape)
    # print(y_pred_12.shape)
    output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
                           '201610': y_pred_10, '201611': y_pred_11, '201612': y_pred_12,
                           '201710': y_pred_10, '201711': y_pred_11, '201712': y_pred_12})

    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    return output

if __name__ == '__main__':
    # Define model
    fc = resnet18_1d(in_channels=1, num_classes=1)
    fc.cuda()

    # actual data
    # preprocessing
    print('preprocessing')

    # train data
    train_df = pd.read_csv("../data/train_2016_v2.csv", parse_dates=["transactiondate"])

    # monthly information
    train_df['transaction_month'] = train_df['transactiondate'].dt.month

    # property information
    prop_df = pd.read_csv("../data/properties_2016.csv")

    # merge property to data
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

    # some manual truncation
    ulimit = np.percentile(train_df.logerror.values, 99)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].loc[train_df['logerror']>ulimit] = ulimit
    train_df['logerror'].loc[train_df['logerror']<llimit] = llimit

    # drop categorical values
    train_y = train_df['logerror'].values
    cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
    train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
    feat_names = train_df.columns

    # fill missing values
    mean_values = train_df.mean(axis=0)
    train_df.fillna(mean_values, inplace=True)

    # make test dataset
    if os.path.isfile(single_results_dir + 'test_df.pkl'):
        print('read from file')
        # read parcelid
        test_df = pd.read_csv("../data/sample_submission.csv")
        test_df.rename(columns={'ParcelId':'parcelid'}, inplace=True)
        test_df = pd.merge(test_df, prop_df, on='parcelid', how='left')
        test_id = test_df['parcelid']

        # read others
        test_df = pd.read_pickle(single_results_dir + 'test_df.pkl')
    else:
        test_df = pd.read_csv("../data/sample_submission.csv")
        test_df.rename(columns={'ParcelId':'parcelid'}, inplace=True)
        test_df = pd.merge(test_df, prop_df, on='parcelid', how='left')
        test_id = test_df['parcelid']
        test_df = test_df.loc[:, feat_names]
        test_df.fillna(mean_values, inplace=True)
        # consider monthly estimation
        test_df_m = []
        for i in range(test_df.shape[0]):
            for m in range(10,13):
                test_tmp = test_df.iloc[i,:]
                test_tmp['transaction_month'] = m
                test_df_m.append(test_tmp)
        test_df = pd.concat(test_df_m, axis=1).T
        test_df.to_pickle(single_results_dir + 'test_df.pkl')
    test_df = test_df.reset_index(drop=True)

    # normalize data
    train_df_mean = train_df.mean()
    train_df_std = train_df.std() + 1.0e-9
    train_y_mean = train_y.mean()
    train_y_std = train_y.std() + 1.0e-9
    train_df_n = (train_df - train_df_mean) / train_df_std
    train_y_n = (train_y - train_y_mean) / train_y_std
    test_df_n = (test_df - train_df_mean) / train_df_std

    train_x_th = torch.from_numpy(train_df_n.values).type(dtype)
    train_y_th = torch.from_numpy(train_y_n).type(dtype)
    train_y_th = train_y_th.view(train_y_th.size()[0], 1)
    train = data_utils.TensorDataset(train_x_th, train_y_th)
    train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True)
    optimizer = optim.Adam(fc.parameters(), lr=1.e-4)

    num_epoch = 10000
    save_epoch = 100
    start_epoch = 0
    single_results_dir = './single_model_results/'
    save_path = 'dnn_results/'
    loss_best = float("Inf")

    # resume
    lt = glob.glob(save_path + 'ckpt_*')
    if lt:
        print('resume')
        lt.sort(key=natural_keys)
        load_status = torch.load(lt[-1])
        start_epoch = load_status['epoch'] - 1
        fc.load_state_dict(load_status['state_dict'])
        model_best = load_status['model_best']
        optimizer.load_state_dict(load_status['optimizer'])

    for ep in range(start_epoch, num_epoch):
        loss_total = 0.
        batch_total = np.ceil(len(train_loader.dataset) / train_loader.batch_size).astype('int')
        _tqdm = dict(total=batch_total)
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader), **_tqdm):
            # Get data
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())

            # Reset gradients
            fc.zero_grad()

            # Forward pass
            batch_x = batch_x.view(batch_x.size()[0],batch_x.size()[1],1)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.view(batch_y.size()[0],batch_y.size()[1],1)
            batch_y = batch_y.permute(0,2,1)
            output = F.smooth_l1_loss(fc(batch_x), batch_y)
            loss = output.data[0]

            # Backward pass
            output.backward()

            # Apply gradients
            optimizer.step()

            loss_total += loss
        # save model
        if (ep + 1) % save_epoch == 0:
            print('model save')
            if loss < loss_best:
                loss_best = loss
                model_best = fc
            save_status = {
                'epoch': ep + 1,
                'state_dict': fc.state_dict(),
                'model_best': model_best,
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(save_status, save_path + 'ckpt_{:08d}.pth.tar'.format(ep))
            print('test data')
            pred_all = np.zeros((test_df_n.shape[0], 1))
            test_batch_size = 30
            batch_total = np.ceil(pred_all.shape[0] / test_batch_size).astype('int')
            _tqdm = dict(total=batch_total)
            for bid in tqdm(range(0, pred_all.shape[0], test_batch_size)):
                test_x_th = torch.from_numpy(test_df_n.values[bid:min(bid+test_batch_size, pred_all.shape[0]),]).type(dtype)
                test_x_th = Variable(test_x_th.cuda())
                test_x_th = test_x_th.view(test_x_th.size()[0], test_x_th.size()[1], 1)
                test_x_th = test_x_th.permute(0,2,1)
                pred = fc(test_x_th).cpu().data.numpy()
                pred_all[bid:min(bid+test_batch_size, pred_all.shape[0])] = pred
            print('pred', pred_all.shape)
            pred_all = np.squeeze(pred_all)
            np.save(single_results_dir + 'resnet_output.npy', pred_all)
            output = make_output_month(pred_all, train_y_mean, train_y_std, test_id)
            output.to_csv('./dnn_results/submission.csv', index=False)

        loss_total /= float(batch_total)
        print('Loss: {:.6f} after {} epochs'.format(loss_total, ep))


