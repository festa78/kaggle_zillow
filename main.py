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
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
    #train_df = train_df.loc[:, (train_df != train_df.ix[0]).any()]
    train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
    feat_names = train_df.columns

    # fill missing values
    mean_values = train_df.mean(axis=0)
    train_df.fillna(mean_values, inplace=True)

    # make test dataset
    test_df = pd.read_csv("../data/sample_submission.csv")
    test_df.rename(columns={'ParcelId':'parcelid'}, inplace=True)
    test_df = pd.merge(test_df, prop_df, on='parcelid', how='left')
    test_id = test_df['parcelid']
    test_df = test_df.loc[:, feat_names]
    test_df.fillna(mean_values, inplace=True)

    # normalize data
    train_df_mean = train_df.mean()
    train_df_std = train_df.std() + 1.0e-9
    train_y_mean = train_y.mean()
    train_y_std = train_y.std() + 1.0e-9
    train_df_n = (train_df - train_df_mean) / train_df_std
    train_y_n = (train_y - train_y_mean) / train_y_std
    test_df_n = (test_df - train_df_mean) / train_df_std

    ## learning
    print('learning')

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
            output = make_output(preds[name], train_y_mean, train_y_std, test_id)
            output.to_csv(single_results_dir + '{}_output.csv'.format(name), index=False)


    # # extreme trees
    # print('extrees')

    # extree_model = ExtraTreesRegressor(n_estimators=10, max_features=32,
    #                                    random_state=0)

    # extree_model.fit(train_df_n.as_matrix(), train_y_n)

    # extree_pred = extree_model.predict(test_df_n)

    # extree_y_pred=[]

    # for i,predict in enumerate(extree_pred):
    #     extree_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
    #     extree_y_pred=np.array(extree_y_pred)

    # extree_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
    #                               '201610': extree_y_pred, '201611': extree_y_pred, '201612': extree_y_pred,
    #                               '201710': extree_y_pred, '201711': extree_y_pred, '201712': extree_y_pred})

    # # set col 'ParceID' to first col
    # cols = extree_output.columns.tolist()
    # cols = cols[-1:] + cols[:-1]
    # extree_output = extree_output[cols]

    # #output
    # extree_output.to_csv('./extree_output.csv', index=False)

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
        dtrain = xgb.DMatrix(train_df_n, train_y_n, feature_names=train_df.columns.values)
        dtest = xgb.DMatrix(test_df_n)
        stime = time.time()
        xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
        print("Time for xgboost fitting: {:03f}".format(time.time() - stime))
        xgb_pred = xgb_model.predict(dtest)
        # xgb_y_pred=[]

        # for i,predict in enumerate(xgb_pred):
        #     xgb_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
        #     xgb_y_pred=np.array(xgb_y_pred)

        # xgb_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
        #                            '201610': xgb_y_pred, '201611': xgb_y_pred, '201612': xgb_y_pred,
        #                            '201710': xgb_y_pred, '201711': xgb_y_pred, '201712': xgb_y_pred})

        # # set col 'ParceID' to first col
        # cols = xgb_output.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # xgb_output = xgb_output[cols]

        #output
        np.save(single_results_dir + 'xgb_output.npy', xgb_pred)
        xgb_output = make_output(xgb_pred, train_y_mean, train_y_std, test_id)
        xgb_output.to_csv(single_results_dir + 'xgb_output.csv', index=False)

    ## ensenble results
    print('ensemble results')
    ens_pred = xgb_pred
    for name, pred in preds.items():
        ens_pred += pred
    ens_pred /= 1 + len(preds)
    # ens_y_pred=[]

    # for i,predict in enumerate(ens_pred):
    #     ens_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
    #     ens_y_pred=np.array(ens_y_pred)

    # ens_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
    #                            '201610': ens_y_pred, '201611': ens_y_pred, '201612': ens_y_pred,
    #                            '201710': ens_y_pred, '201711': ens_y_pred, '201712': ens_y_pred})

    # # set col 'ParceID' to first col
    # cols = ens_output.columns.tolist()
    # cols = cols[-1:] + cols[:-1]
    # ens_output = ens_output[cols]

    #output
    ens_output = make_output(ens_pred, train_y_mean, train_y_std, test_id)
    ens_output.to_csv('../results/submission_ensenble.csv', index=False)
