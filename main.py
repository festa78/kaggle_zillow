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

# fill missing values
mean_values = train_df.mean(axis=0)

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
feat_names = train_df.columns.values

# make test dataset
test_df = pd.read_csv("../data/sample_submission.csv")
test_df.rename(columns={'ParcelId':'parcelid'}, inplace=True)
test_df = pd.merge(test_df, prop_df, on='parcelid', how='left')
test_df.fillna(mean_values, inplace=True)
test_df = test_df.drop(['201610', '201611', '201612', '201710', '201711', '201712'], axis=1)
test_id = test_df['parcelid']
test_df = test_df.drop(['parcelid'] + cat_cols, axis=1)

# normalize data
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_y_mean = train_y.mean()
train_y_std = train_y.std()
train_df_n = (train_df - train_df_mean) / train_df_std
train_y_n = (train_y - train_y_mean) / train_y_std
test_df_n = (test_df - train_df_mean) / train_df_std

## learning
print('learning')
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

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

## inference
print('inference')
xgb_pred = model.predict(dtest)

y_pred=[]

for i,predict in enumerate(xgb_pred):
    y_pred.append(str(round(predict * train_y_std + train_y_mean,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': test_id.astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

#output
output.to_csv('../results/submission_normalize.csv', index=False)
