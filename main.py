#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
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
#from sklearn.linear_model import RidgeCV

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
    ulimit = np.percentile(train_df.logerror.values, 99)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].loc[train_df['logerror']>ulimit] = ulimit
    train_df['logerror'].loc[train_df['logerror']<llimit] = llimit

    # col = "finishedsquarefeet12"
    # ulimit = np.percentile(train_df[col].values, 99.5)
    # llimit = np.percentile(train_df[col].values, 0.5)
    # train_df[col].loc[train_df[col]>ulimit] = ulimit
    # train_df[col].loc[train_df[col]<llimit] = llimit

    # col = "calculatedfinishedsquarefeet"
    # ulimit = np.percentile(train_df[col].values, 99.5)
    # llimit = np.percentile(train_df[col].values, 0.5)
    # train_df[col].loc[train_df[col]>ulimit] = ulimit
    # train_df[col].loc[train_df[col]<llimit] = llimit

    # col = "taxamount"
    # ulimit = np.percentile(train_df[col].values, 99.5)
    # llimit = np.percentile(train_df[col].values, 0.5)
    # train_df[col].loc[train_df[col]>ulimit] = ulimit
    # train_df[col].loc[train_df[col]<llimit] = llimit

    # drop categorical values
    train_y = train_df['logerror'].values
    cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    #train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
    #train_df = train_df.loc[:, (train_df != train_df.ix[0]).any()]
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

    # stacking approach
    print('train data stacking')
    if os.path.isfile(single_results_dir + 'train_features_output.npy'):
        print('read from file')
        train_features = np.load(single_results_dir + 'train_features_output.npy')
    else:
        # split data
        rs = KFold(n_splits=5)
        # rid = np.random.permutation(train_df.shape[0])
        # split_id = [
        #     (rid[int(len(rid) / 2):], rid[:int(len(rid) / 2)]),
        #     (rid[:int(len(rid) / 2)], rid[int(len(rid) / 2):])
        # ]

        # feature extraction
        train_features = np.zeros([train_df_n.shape[0], len(ESTIMATORS) + 1])
        # for train_id_s, test_id_s in split_id:
        for train_id_s, test_id_s in rs.split(train_df_n):
            train_df_s = train_df_n.iloc[train_id_s,:]
            train_y_s = train_y_n[train_id_s]
            test_df_s = train_df_n.iloc[test_id_s,:]

            preds_s = []
            for name, estimator in ESTIMATORS.items():
                print(name)
                stime = time.time()
                estimator.fit(train_df_s.as_matrix(), train_y_s)
                print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
                pred = estimator.predict(test_df_s)
                preds_s.append(pred)

            # xgboost
            print('xgboost')
            xgb_params = {
                'eta': 0.05,
                'max_depth': 8,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'objective': 'reg:linear',
                'silent': 1,
                'seed' : 0
            }
            dtrain = xgb.DMatrix(train_df_s, train_y_s, feature_names=train_df.columns.values)
            dtest = xgb.DMatrix(test_df_s)
            stime = time.time()
            xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
            print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
            preds_s.append(xgb_model.predict(dtest))

            preds_s = np.stack(preds_s, axis=1)
            train_features[test_id_s,:] = preds_s
            # for idx in range(preds_s.shape[0]):
            #     train_features.append(preds_s[idx,:])

        # train_features = np.stack(train_features, axis=0)
        # train_features = train_features[np.argsort(rid),:]
        np.save(single_results_dir + 'train_features_output.npy', train_features)
    train_features = pd.DataFrame(train_features, columns=np.arange(0,-train_features.shape[1], -1))

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
            estimator.fit(train_df_n.as_matrix(), train_y_n)
            print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
            pred = estimator.predict(test_df_n)
            preds_s.append(pred)

        # xgboost
        print('xgboost')
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'silent': 1,
            'seed' : 0
        }
        dtrain = xgb.DMatrix(train_df_n, train_y_n)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
        print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
        pred = xgb_model.predict(dtest)
        preds_s.append(pred)

        preds_s = np.stack(preds_s, axis=1)
        for idx in range(preds_s.shape[0]):
            test_features.append(preds_s[idx,:])
        test_features = np.stack(test_features, axis=0)
        np.save(single_results_dir + 'test_features_output.npy', test_features)
    test_features = pd.DataFrame(test_features, columns=np.arange(0,-test_features.shape[1], -1))

    ## learning
    print('learning')
    # train_df_n = pd.concat([train_df_n, train_features], axis=1)
    # test_df_n = pd.concat([test_df_n, test_features], axis=1)
    #train_df_n = train_features
    #test_df_n = test_features
    preds = dict()
    for name, estimator in ESTIMATORS.items():
        print(name)
        if os.path.isfile(single_results_dir + '{}_output.npy'.format(name)):
            print('read from file')
            preds[name] = np.load(single_results_dir + '{}_output.npy'.format(name))
        else:
            stime = time.time()
            estimator.fit(train_df_n.as_matrix(), train_y_n)
            print("Time for {} fitting: {:03f}".format(name, time.time() - stime))
            preds[name] = estimator.predict(test_df_n)

            #output
            np.save(single_results_dir + '{}_output.npy'.format(name), preds[name])
            #output = make_output(preds[name], train_y_mean, train_y_std, test_id)
            output = make_output_month(preds[name], train_y_mean, train_y_std, test_id)
            output.to_csv(single_results_dir + '{}_output.csv'.format(name), index=False)

    # xgboost
    print('xgboost')
    if os.path.isfile(single_results_dir + 'xgb_output.npy'):
        print('read from file')
        xgb_pred = np.load(single_results_dir + 'xgb_output.npy'.format(name))
    else:
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'silent': 1,
            'seed' : 0
        }
        dtrain = xgb.DMatrix(train_df_n, train_y_n)
        # dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df_n.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
        print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
        xgb_pred = xgb_model.predict(dtest)

        #output
        np.save(single_results_dir + 'xgb_output.npy', xgb_pred)
        #xgb_output = make_output(xgb_pred, train_y_mean, train_y_std, test_id)
        xgb_output = make_output_month(xgb_pred, train_y_mean, train_y_std, test_id)
        xgb_output.to_csv(single_results_dir + 'xgb_output.csv', index=False)

    ## resnet results
    print('resnet')
    print('read from file')
    resnet_pred = np.load(single_results_dir + 'resnet_output.npy')

    ## ensenble results
    print('ensemble results')
    ens_pred = xgb_pred
    ens_pred += resnet_pred
    for name, pred in preds.items():
        ens_pred += pred
    ens_pred /= 2 + len(preds)

    #output
    #ens_output = make_output(ens_pred, train_y_mean, train_y_std, test_id)
    ens_output = make_output_month(ens_pred, train_y_mean, train_y_std, test_id)
    ens_output.to_csv('../results/submission_ensenble.csv', index=False)
