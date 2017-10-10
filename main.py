#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import glob
import re
import math
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.models as models
from torch.autograd import Variable
import lightgbm as lgb

#from sklearn.linear_model import RidgeCV
from main_dnn import resnet18_1d
from main_dnn import natural_keys


color = sns.color_palette()
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
single_results_dir = './single_model_results/'

def make_output(pred, train_y_mean, train_y_std, test_id):
    y_pred = []
    for i,predict in enumerate(pred):
        y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
    y_pred=np.array(y_pred)

    output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
                           '201610': y_pred, '201611': y_pred, '201612': y_pred,
                           '201710': y_pred, '201711': y_pred, '201712': y_pred})

    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    return output

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


def resnet_learn(train_loader, model, optimizer, basename):
    num_epoch = 5000
    save_epoch = 100
    start_epoch = 0
    save_path = 'dnn_results/'
    loss_best = float("Inf")
    # resume
    lt = glob.glob(save_path + '{}_ckpt_*'.format(basename))
    if lt:
        print('resume')
        lt.sort(key=natural_keys)
        load_status = torch.load(lt[-1])
        start_epoch = load_status['epoch'] - 1
        model.load_state_dict(load_status['state_dict'])
        model_best = load_status['model_best']
        optimizer.load_state_dict(load_status['optimizer'])

    for ep in range(start_epoch, num_epoch):
        loss_total = 0.
        batch_total = np.ceil(len(train_loader.dataset) / train_loader.batch_size).astype('int')
        _tqdm = dict(total=batch_total)

        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader), **_tqdm):
            # Get data
            batch_x = Variable(batch_x.type(torch.FloatTensor).cuda())
            batch_y = Variable(batch_y.type(torch.FloatTensor).cuda())

            # Reset gradients
            model.zero_grad()

            # Forward pass
            batch_x = batch_x.view(batch_x.size()[0],batch_x.size()[1],1)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.view(batch_y.size()[0],batch_y.size()[1],1)
            batch_y = batch_y.permute(0,2,1)
            output = F.smooth_l1_loss(model(batch_x), batch_y)
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
                model_best = model
            save_status = {
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'model_best': model_best,
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(save_status, save_path + '{}_ckpt_{:08d}.pth.tar'.format(basename, ep))

    return model_best


def resnet_pred(model, test_df_n):
    pred_all = np.zeros((test_df_n.shape[0], 1))
    test_batch_size = 30
    batch_total = np.ceil(pred_all.shape[0] / test_batch_size).astype('int')
    _tqdm = dict(total=batch_total)
    for bid in tqdm(range(0, pred_all.shape[0], test_batch_size)):
        test_x_th = torch.from_numpy(test_df_n.values[bid:min(bid+test_batch_size, pred_all.shape[0]),]).type(torch.FloatTensor)
        test_x_th = Variable(test_x_th.cuda())
        test_x_th = test_x_th.view(test_x_th.size()[0], test_x_th.size()[1], 1)
        test_x_th = test_x_th.permute(0,2,1)
        pred = model(test_x_th).cpu().data.numpy()
        pred_all[bid:min(bid+test_batch_size, pred_all.shape[0])] = pred
    pred_all = np.squeeze(pred_all)
    print('pred', pred_all.shape)
    return pred_all


if __name__ == "__main__":
    ## preprocessing
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
    # ulimit = np.percentile(train_df.logerror.values, 99)
    # llimit = np.percentile(train_df.logerror.values, 1)
    ulimit = 0.419
    llimit = -0.4
    train_df['logerror'].loc[train_df['logerror']>ulimit] = ulimit
    train_df['logerror'].loc[train_df['logerror']<llimit] = llimit

    # drop categorical values
    train_y = train_df['logerror'].values
    cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    #cat_cols = ["propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    #train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
    #train_df = train_df.loc[:, (train_df != train_df.ix[0]).any()]
    #train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
    feat_names = train_df.columns

    # fill missing values
    mean_values = train_df.median(axis=0)
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

    print(train_df.shape)
    print(test_df.shape)
    # normalize data
    train_df_mean = train_df.mean()
    train_df_std = train_df.std() + 1.0e-9
    train_y_mean = train_y.mean()
    train_y_std = train_y.std() + 1.0e-9
    train_df_n = (train_df - train_df_mean) / train_df_std
    train_y_n = (train_y - train_y_mean) / train_y_std
    test_df_n = (test_df - train_df_mean) / train_df_std

    # Fit estimators
    ESTIMATORS = {
        "adaboost": AdaBoostRegressor(),
        "gradient_boost": GradientBoostingRegressor(),
        "knn": KNeighborsRegressor(n_jobs=2),
        "svr": SVR(C=1.0, epsilon=0.2, verbose=True),
        "extra_trees": ExtraTreesRegressor(),
        "random_forest": RandomForestRegressor(),
        "linear_regression": LinearRegression(),
        #"gpr": GaussianProcessRegressor(),
        #"kernel_ridge": KernelRidge(alpha=1.0),
    }

    lightgbm_params = {
        'max_bin': 10,
        'learning_rate': 0.0021,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l1',
        'sub_feature': 0.345,
        'bagging_fraction': 0.85,
        'bagging_freq': 40,
        'num_leaves': 512,
        'min_data': 10,
        'min_hessian': 0.05,
        'verbose': 0,
        'feature_fraction_seed': 2,
        'bagging_seed': 2,
    }

    xgb_params1 = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'silent': 1,
        'seed' : 0
    }

    xgb_params2 = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'seed' : 0
    }

    # stacking approach
    print('train data stacking')
    if os.path.isfile(single_results_dir + 'train_features_output.npy'):
        print('read from file')
        train_features = np.load(single_results_dir + 'train_features_output.npy')
    else:
        # split data
        rs = KFold(n_splits=5)

        # feature extraction
        train_features = np.zeros([train_df_n.shape[0], len(ESTIMATORS) + 3])
        count = 0
        # for train_id_s, test_id_s in split_id:
        for train_id_s, test_id_s in rs.split(train_df_n):
            train_df_s = train_df_n.iloc[train_id_s,:]
            train_y_s = train_y_n[train_id_s]
            test_df_s = train_df_n.iloc[test_id_s,:]

            preds_s = []

            for name, estimator in ESTIMATORS.items():
                print(name)
                stime = time.time()
                estimator.fit(train_df_s.values, train_y_s)
                print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
                pred = estimator.predict(test_df_s.values)
                preds_s.append(pred)

            # lightgbm
            print('lightgbm')
            dtrain = lgb.Dataset(train_df_s.values, label=train_y_s)
            stime = time.time()
            lgbm_model = lgb.train(lightgbm_params, dtrain, 430)
            print("Time for lightgbm fitting: {:03f}".format(time.time() - stime))
            preds_s.append(lgbm_model.predict(test_df_s.values))

            # xgboost
            print('xgboost1')
            dtrain = xgb.DMatrix(train_df_s, train_y_s, feature_names=train_df.columns.values)
            dtest = xgb.DMatrix(test_df_s)
            stime = time.time()
            xgb_model = xgb.train(dict(xgb_params1), dtrain, num_boost_round=250)
            print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
            preds_s.append(xgb_model.predict(dtest))

            print('xgboost2')
            dtrain = xgb.DMatrix(train_df_s, train_y_s, feature_names=train_df.columns.values)
            dtest = xgb.DMatrix(test_df_s)
            stime = time.time()
            xgb_model = xgb.train(dict(xgb_params2), dtrain, num_boost_round=150)
            print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
            preds_s.append(xgb_model.predict(dtest))

            # resnet
            # print('resnet')
            # # pytorch format
            # train_x_th = torch.from_numpy(train_df_s.values).type(torch.FloatTensor)
            # train_y_th = torch.from_numpy(train_y_s).type(torch.FloatTensor)
            # train_y_th = train_y_th.view(train_y_th.size()[0], 1)
            # train_y_th = torch.from_numpy(train_y_s[:,np.newaxis])
            # train = data_utils.TensorDataset(train_x_th, train_y_th)
            # train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True)

            # model = resnet18_1d(in_channels=1, num_classes=1)
            # model.cuda()
            # optimizer = optim.Adam(model.parameters(), lr=1.e-4)
            # model = resnet_learn(train_loader, model, optimizer, 'train_feature_{:d}'.format(count))
            # count += 1
            # preds_s.append(resnet_pred(model, test_df_s))

            preds_s = np.stack(preds_s, axis=1)
            train_features[test_id_s,:] = preds_s

        # train_features = np.stack(train_features, axis=0)
        # train_features = train_features[np.argsort(rid),:]
        np.save(single_results_dir + 'train_features_output.npy', train_features)
    train_features = pd.DataFrame(train_features, columns=np.arange(0,-train_features.shape[1], -1))

    # linear regression of ensemble weight parameters
    lr = LinearRegression()
    lr.fit(train_features.values, train_y_n)
    ens_weight = lr.coef_
    print('feature size', train_features.shape)
    print('ens_weight size', ens_weight.shape[0])
    assert(ens_weight.shape[0] == len(ESTIMATORS) + 3)

    print('test data stacking')
    if os.path.isfile(single_results_dir + 'test_features_output.npy'):
        print('read from file')
        test_features = np.load(single_results_dir + 'test_features_output.npy')
    else:
        test_features = []
        preds_s = []
        for name, estimator in ESTIMATORS.items():
            print(name)
            stime = time.time()
            estimator.fit(train_df_n.values, train_y_n)
            print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
            pred = estimator.predict(test_df_n.values)
            preds_s.append(pred)

        # lightgbm
        print('lightgbm')
        dtrain = lgb.Dataset(train_df_n.values, label=train_y_n)
        stime = time.time()
        lgbm_model = lgb.train(lightgbm_params, dtrain, 430)
        print("Time for lightgbm fitting: {:03f}".format(time.time() - stime))
        preds_s.append(lgbm_model.predict(test_df_n.values))

        # xgboost
        print('xgboost1')
        dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb_model = xgb.train(dict(xgb_params1), dtrain, num_boost_round=50)
        print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
        preds_s.append(xgb_model.predict(dtest))

        print('xgboost2')
        dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb_model = xgb.train(dict(xgb_params2), dtrain, num_boost_round=50)
        print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
        preds_s.append(xgb_model.predict(dtest))

        # resnet
        # print('resnet')
        # # pytorch format
        # train_x_th = torch.from_numpy(train_df_n.values).type(torch.FloatTensor)
        # train_y_th = torch.from_numpy(train_y_n).type(torch.FloatTensor)
        # train_y_th = train_y_th.view(train_y_th.size()[0], 1)
        # train_y_th = torch.from_numpy(train_y_n[:,np.newaxis])
        # train = data_utils.TensorDataset(train_x_th, train_y_th)
        # train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True)

        # model = resnet18_1d(in_channels=1, num_classes=1)
        # model.cuda()
        # optimizer = optim.Adam(model.parameters(), lr=1.e-4)
        # model = resnet_learn(train_loader, model, optimizer, 'test_feature')
        # preds_s.append(resnet_pred(model, test_df_n))

        preds_s = np.stack(preds_s, axis=1)
        for idx in range(preds_s.shape[0]):
            test_features.append(preds_s[idx,:])
        test_features = np.stack(test_features, axis=0)
        np.save(single_results_dir + 'test_features_output.npy', test_features)
    test_features = pd.DataFrame(test_features, columns=np.arange(0,-test_features.shape[1], -1))

    ## learning
    print('learning')
    train_df_n = pd.concat([train_df_n, train_features], axis=1)
    test_df_n = pd.concat([test_df_n, test_features], axis=1)
    # train_df_n = train_features
    # test_df_n = test_features
    preds = dict()
    for name, estimator in ESTIMATORS.items():
        print(name)
        if os.path.isfile(single_results_dir + '{}_output.npy'.format(name)):
            print('read from file')
            preds[name] = np.load(single_results_dir + '{}_output.npy'.format(name))
        else:
            stime = time.time()
            estimator.fit(train_df_n.values, train_y_n)
            print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
            preds[name] = estimator.predict(test_df_n.values)

            #output
            np.save(single_results_dir + '{}_output.npy'.format(name), preds[name])
            #output = make_output(preds[name], train_y_mean, train_y_std, test_id)
            output = make_output_month(preds[name], train_y_mean, train_y_std, test_id)
            output.to_csv(single_results_dir + '{}_output.csv'.format(name), index=False)

    print('lightgbm')
    if os.path.isfile(single_results_dir + 'lightgbm_output.npy'):
        print('read from file')
        lgbm_pred = np.load(single_results_dir + 'lightgbm_output.npy'.format(name))
    else:
        dtrain = lgb.Dataset(train_df_n.values, label=train_y_n)
        stime = time.time()
        lgbm_model = lgb.train(lightgbm_params, dtrain, 430)
        print("Time for lightgbm fitting: {:03f}".format(time.time() - stime))
        lgbm_pred = lgbm_model.predict(test_df_n.values)

        #output
        np.save(single_results_dir + 'lgbm_output.npy', lgbm_pred)
        lgbm_output = make_output_month(lgbm_pred, train_y_mean, train_y_std, test_id)
        lgbm_output.to_csv(single_results_dir + 'lgbm_output.csv', index=False)

    # xgboost
    print('xgboost1')
    if os.path.isfile(single_results_dir + 'xgb1_output.npy'):
        print('read from file')
        xgb1_pred = np.load(single_results_dir + 'xgb1_output.npy'.format(name))
    else:
        dtrain = xgb.DMatrix(train_df_n, train_y_n)
        # dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df_n.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb1_model = xgb.train(dict(xgb_params1, silent=0), dtrain, num_boost_round=50)
        print("Time for xgboost1 fitting: {:03f}".format(time.time() - stime))
        xgb1_pred = xgb1_model.predict(dtest)

        #output
        np.save(single_results_dir + 'xgb1_output.npy', xgb1_pred)
        #xgb_output = make_output(xgb_pred, train_y_mean, train_y_std, test_id)
        xgb1_output = make_output_month(xgb1_pred, train_y_mean, train_y_std, test_id)
        xgb1_output.to_csv(single_results_dir + 'xgb1_output.csv', index=False)

    print('xgboost2')
    if os.path.isfile(single_results_dir + 'xgb2_output.npy'):
        print('read from file')
        xgb2_pred = np.load(single_results_dir + 'xgb2_output.npy'.format(name))
    else:
        dtrain = xgb.DMatrix(train_df_n, train_y_n)
        # dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df_n.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb2_model = xgb.train(dict(xgb_params1, silent=0), dtrain, num_boost_round=50)
        print("Time for xgboost2 fitting: {:03f}".format(time.time() - stime))
        xgb2_pred = xgb2_model.predict(dtest)

        #output
        np.save(single_results_dir + 'xgb2_output.npy', xgb2_pred)
        #xgb_output = make_output(xgb_pred, train_y_mean, train_y_std, test_id)
        xgb2_output = make_output_month(xgb2_pred, train_y_mean, train_y_std, test_id)
        xgb2_output.to_csv(single_results_dir + 'xgb2_output.csv', index=False)

    ## resnet results
    # print('resnet')
    # if os.path.isfile(single_results_dir + 'resnet_output.npy'):
    #     print('read from file')
    #     resnet_pred = np.load(single_results_dir + 'resnet_output.npy'.format(name))
    # else:
    #     # pytorch format
    #     train_x_th = torch.from_numpy(train_df_n.values).type(torch.FloatTensor)
    #     train_y_th = torch.from_numpy(train_y_n).type(torch.FloatTensor)
    #     train_y_th = train_y_th.view(train_y_th.size()[0], 1)
    #     train_y_th = torch.from_numpy(train_y_n[:,np.newaxis])
    #     train = data_utils.TensorDataset(train_x_th, train_y_th)
    #     train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True)

    #     model = resnet18_1d(in_channels=1, num_classes=1)
    #     model.cuda()
    #     optimizer = optim.Adam(model.parameters(), lr=1.e-4)
    #     model = resnet_learn(train_loader, model, optimizer, 'prediction')
    #     resnet_predict = (resnet_pred(model, test_df_n))
    #     np.save(single_results_dir + 'resnet_output.npy', resnet_predict)
    #     resnet_output = make_output_month(resnet_predict, train_y_mean, train_y_std, test_id)
    #     resnet_output.to_csv(single_results_dir + 'resnet_output.csv', index=False)

    ## ensenble results
    print('ensemble results')
    count = -3
    ens_pred = lgbm_pred * ens_weight[count]
    count += 1
    ens_pred = xgb1_pred * ens_weight[count]
    count += 1
    ens_pred = xgb2_pred * ens_weight[count]
    # ens_pred += resnet_predict
    for name, pred in preds.items():
        count += 1
        ens_pred += pred * ens_weight[count]
    # ens_pred /= 1 + len(preds)

    #output
    #ens_output = make_output(ens_pred, train_y_mean, train_y_std, test_id)
    ens_output = make_output_month(ens_pred, train_y_mean, train_y_std, test_id)
    ens_output.to_csv('../results/submission_ensenble.csv', index=False)
