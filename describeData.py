# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:12:23 2018

@author: Administrator
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#pd_train1=pd.read_csv('train_1.csv')
#pd.to_datetime(pd_train1['时间'])
#print(pd_train1.columns)
#pd_windspeed=pd_train1['实际功率']
##matplotlib inline
#pd_windspeed.plot()

pd_train1=pd.read_csv('train_1.csv')
pd_train1['时间'] = pd.to_datetime(pd_train1['时间'])
pd_oneday=pd_train1[pd_train1['时间']<='2016-04-05 23:59:59']
pd_oneday.drop(columns=['时间', '风速','压强', '湿度'], inplace=True)
pd_trans = MinMaxScaler().fit_transform(pd_oneday)
pd_dataframe=pd.DataFrame(pd_trans,columns=['fs', 'wd','fzd','sffzd', 'sjgl'])
#matplotlib inline
pd_dataframe.plot()