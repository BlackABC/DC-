# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:42:16 2018

@author: Administrator
"""

print("start......")


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from lightgbm import LGBMRegressor, plot_importance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

pd_train = pd.read_csv('train_4.csv')
pd_train['时间']=pd.to_datetime(pd_train['时间'])
pd_train['hour']=pd_train['时间'].dt.hour
pd_train['month']=pd_train['时间'].dt.month

pd_train.drop(columns=['时间', '实发辐照度'], inplace=True)
Y_train = pd_train['实际功率']
X_train=pd_train.drop(columns=['实际功率'])

feature_trans=PolynomialFeatures(degree=2, include_bias=False)
X_train1=pd.DataFrame(feature_trans.fit_transform(X_train[['风速', '风向']]))
X_train1.drop(columns=[0, 1], inplace = True)
X_train1.rename(columns={2:'w1', 3:'w2', 4:'w3'}, inplace=True)
X_train=pd.concat([X_train, X_train1], axis=1)

X_train2=pd.DataFrame(feature_trans.fit_transform(X_train[['风速', '温度', '压强', '湿度']]))
X_train2.drop(columns=[0, 1, 2, 3], inplace=True)
X_train2.rename(columns={4:'t1', 5:'t2', 6:'t3', 7:'t4',
                         8:'t5', 9:'t6', 10:'t7', 11:'t8',
                         12:'t9', 13:'t10'}, inplace = True)
X_train = pd.concat([X_train, X_train2], axis=1)

X_train = MinMaxScaler().fit_transform(X_train)

lgb=LGBMRegressor()
lgb.fit(X_train, Y_train)

# test code
pd_test=pd.read_csv("test_4.csv")
pd_test['时间']=pd.to_datetime(pd_test['时间'])
pd_test['hour']=pd_test['时间'].dt.hour
pd_test['month']=pd_test['时间'].dt.month

pd_test.drop(columns=['id', '时间'], inplace=True)
X_test = pd_test

feature_trans=PolynomialFeatures(degree=2, include_bias=False)
X_test1=pd.DataFrame(feature_trans.fit_transform(X_test[['风速', '风向']]))
X_test1.drop(columns=[0, 1], inplace = True)
X_test1.rename(columns={2:'w1', 3:'w2', 4:'w3'}, inplace=True)
X_test = pd.concat([X_test, X_test1], axis=1)

X_test2=pd.DataFrame(feature_trans.fit_transform(X_test[['风速', '温度', '压强', '湿度']]))
X_test2.drop(columns=[0, 1, 2, 3], inplace=True)
X_test2.rename(columns={4:'t1', 5:'t2', 6:'t3', 7:'t4',
                         8:'t5', 9:'t6', 10:'t7', 11:'t8',
                         12:'t9', 13:'t10'}, inplace = True)
X_test = pd.concat([X_test, X_test2], axis=1)

X_test = MinMaxScaler().fit_transform(X_test)

Y_pred= lgb.predict(X_test)

Y_pred = Y_pred.tolist()
Y_pred = pd.DataFrame(Y_pred)
Y_pred.rename(columns={0:"predicition"}, inplace = True)
Y_pred.to_csv("result4.csv", index = False)