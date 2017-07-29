#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

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
train_df = train_df.loc[:, (train_df != train_df.ix[0]).any()]
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

# extreme trees
print('extrees')
from sklearn.ensemble import ExtraTreesRegressor

extree_model = ExtraTreesRegressor(n_estimators=10, max_features=32,
                                   random_state=0)

extree_model.fit(train_df_n.as_matrix(), train_y_n)

extree_pred = extree_model.predict(test_df_n)

extree_y_pred=[]

for i,predict in enumerate(extree_pred):
    extree_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
extree_y_pred=np.array(extree_y_pred)

extree_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
                           '201610': extree_y_pred, '201611': extree_y_pred, '201612': extree_y_pred,
                           '201710': extree_y_pred, '201711': extree_y_pred, '201712': extree_y_pred})

# set col 'ParceID' to first col
cols = extree_output.columns.tolist()
cols = cols[-1:] + cols[:-1]
extree_output = extree_output[cols]

#output
extree_output.to_csv('./extree_output.csv', index=False)

# xgboost
print('xgboost')
import xgboost as xgb
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

xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

xgb_pred = xgb_model.predict(dtest)

xgb_y_pred=[]

for i,predict in enumerate(xgb_pred):
    xgb_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
xgb_y_pred=np.array(xgb_y_pred)

xgb_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
                           '201610': xgb_y_pred, '201611': xgb_y_pred, '201612': xgb_y_pred,
                           '201710': xgb_y_pred, '201711': xgb_y_pred, '201712': xgb_y_pred})

# set col 'ParceID' to first col
cols = xgb_output.columns.tolist()
cols = cols[-1:] + cols[:-1]
xgb_output = xgb_output[cols]

#output
xgb_output.to_csv('./xgb_output.csv', index=False)

## ensenble results
print('ensemble results')
ens_pred = 0.5 * xgb_pred + 0.5 * extree_pred

ens_y_pred=[]

for i,predict in enumerate(ens_pred):
    ens_y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
ens_y_pred=np.array(ens_y_pred)

ens_output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
                           '201610': ens_y_pred, '201611': ens_y_pred, '201612': ens_y_pred,
                           '201710': ens_y_pred, '201711': ens_y_pred, '201712': ens_y_pred})

# set col 'ParceID' to first col
cols = ens_output.columns.tolist()
cols = cols[-1:] + cols[:-1]
ens_output = ens_output[cols]

#output
ens_output.to_csv('../results/submission_ensenble.csv', index=False)
