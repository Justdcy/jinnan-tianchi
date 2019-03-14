
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 20:53:15 2019

@author: dell
"""

import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)

train_path = './data/jinnan_round1_train_20181227.csv'
testA_path = './data/jinnan_round1_testA_20181227.csv'
testB_path = './data/jinnan_round1_testB_20190121.csv'
testC_path = './data/jinnan_round1_test_20190201.csv'

testA_ans_path = './data/jinnan_round1_ansA_20190125.csv'
testB_ans_path = './data/jinnan_round1_ansB_20190125.csv'
testC_ans_path = './data/jinnan_round1_ans_20190201.csv'
test_path = './data/FuSai.csv'
test_optimize_path = './data/optimize.csv'

train1  = pd.read_csv(train_path, encoding = 'gb18030')
testA  = pd.read_csv(testA_path, encoding = 'gb18030')
testB  = pd.read_csv(testB_path, encoding = 'gb18030')
testC  = pd.read_csv(testC_path, encoding = 'gb18030')
test  = pd.read_csv(test_path, encoding = 'gb18030')
test_optimize  = pd.read_csv(test_optimize_path, encoding = 'gb18030')

testA_ans  = pd.read_csv(testA_ans_path, encoding = 'gb18030',header=None)
testB_ans  = pd.read_csv(testB_ans_path, encoding = 'gb18030',header=None)
testC_ans  = pd.read_csv(testC_ans_path, encoding = 'gb18030',header=None)

testA['收率'] = testA_ans[1]
testB['收率'] = testB_ans[1]
testC['收率'] = testC_ans[1]

train_all = pd.concat([train1,testA,testB,testC],axis=0,ignore_index=True)
train = train_all

min_threshold = 0.85
max_threshold = 1.00

####################time_model需要的函数#########################
def timeTranSecond(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    
    return tm

def getDuration(se):
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        return np.nan 
        
    try:
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    except:
        if se=='19:-20:05':
            return 1
        elif se=='15:00-1600':
            return 1
    
    return tm

#############period_model需要的函数#########################
    
###将A5复制给A7，并将A8中缺失值赋为0
def copy_time(df_train):

    df_train = df_train.fillna({'A2':0})#A2添0
    df_train = df_train.fillna({'A3':0})#A3用0
    i1=[]
    i2=[]
    df_train_A8 = df_train['A8']
    for i in df_train.index:
      #  df_train.at[i,'A8'] = 0.0
        if (df_train_A8[i]>0):
        #while(df_train['A8']>0)
            i1.append(i)
        else:
            i2.append(i)
    
    for j in i2:
        df_train['A8'][j] = 0.0
    
    for j in i2:
        df_train['A7'][j] = df_train['A5'][j]
      
    return df_train
    
#将时间段分开为前后两列的子函数
def get_two_time(se):
    try:
        s,e=se.split('-')
    except:      ##如果时间段为缺失值或其他时
        s= 'nan:nan'
        e='nan:nan'
    return s,e

###将时间段分开为前后两列
def Duration_split(f,f_index,df_train):
    temp_1 = []
    temp_2 = []
    f_1 = f + '_1'
    f_2 = f + '_2'
    
    df_train_temp = df_train[f]
    for ii in range(len(df_train_temp)):
        t_1,t_2 = get_two_time(df_train_temp[ii])
        temp_1.append(t_1)
        temp_2.append(t_2)    
        
    #在f_index列数插入名字为f_1的列，该列值为temp_1
    df_train.insert(f_index,f_1,temp_1)
    df_train.insert(f_index+1,f_2,temp_2)
    df_train.drop(f,axis=1, inplace=True)   ###删除原本列
    
    return df_train

###
def get_columns(f,df_train):
    col_name = df_train.columns.tolist()

    for j in range(len(col_name)):
        if (col_name[j] == f):
            f_index = j
    return f_index
    
# 日期中有些输入错误和遗漏:
def t2s(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600
        elif t=='1900/1/1 2:30':
            return 2*3600+30*60
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = int(t)*3600+int(m)*60+int(s)
    except:
        return 30
    
    return (tm/60)

# 日期中有些输入错误和遗漏
def t2s1(se):
    try:
        t,m=re.split("[:,;]",se)
    except:
        if se=='14::30':
            return 14*3600+30*60
        elif se=='22"00':
            return 22*3600
        elif se=='1600':
            return 16*3600
        elif se=='1900/3/10 0':
            return -1
        elif se=='nan':
            return 'nan'
        elif se==-1:
            return -1
        else:
            return 0
    
    try:
        tm = int(t)*3600+int(m)*60
    except:
        return 'nan'
    
    return (tm/60)


###后时间减去前面的时间，并处理缺失值
def interval_missing(df_train):
    
    for f in ['A20_1','A20_2','A28_1','A28_2','B4_1','B4_2','B9_1','B9_2','B10_1','B10_2','B11_1','B11_2']:
        df_train[f] = df_train[f].apply(t2s1)

    #f = ['A5','A7','A9','A11','A14','A16','A20_1','A20_2','A24','A26','A28_1','A28_2','B4_1','B4_2','B5','B7','B9_1','B9_2','B10_1','B10_2','B11_1','B11_2']
    f = ['B11_2','B11_1','B10_2','B10_1','B9_2','B9_1','B7','B5','B4_2','B4_1','A28_2','A28_1','A26','A24','A20_2','A20_1','A16','A14','A11','A9','A7','A5']
    #f = ['B11_2','B11_1']
    
    for i in range(len(f)-1):
        f2 = f[i]
        f1 = f[i+1]
    
        time_2 = df_train[f2]
        time_1 = df_train[f1]
        time_temp = []
        
        for j in range(len(time_2)):
            if time_2[j] == 'nan':
                time_temp.append(0)
            else:
                interval = float(time_2[j]) - float(time_1[j])
                if interval<0:
                    interval = interval+24*60
                time_temp.append(interval)
    
        df_train[f2] = time_temp

    f_abnormal = ['A9','A11','A14','A28_1','A28_2','B4_1','B4_2','B5']
    
    for f in f_abnormal:
        df_train_temp = df_train[f]
        df_train_mean = int(df_train[f].mean())
        for i in range(len(df_train_temp)):
            if np.abs(df_train_temp[i])>10*np.abs(df_train_mean):
                df_train_temp[i] = np.abs(df_train_mean)
    
        df_train[f] = df_train_temp

    df_train_temp = df_train['A25']
    df_train_temp[1304] = 75

    f_null_m = ['A20_2','A21','A23','A24','A25','A27','B1','B2','B3','B4_1','B4_2','B8','B10_1','B10_2','B12','B13']
    f_null_0 = ['B11_1','B11_2']
    
    for f in f_null_m:
        df_train_median = int(df_train[f].median())
        df_train=df_train.fillna({f:df_train_median})
    
    for f in f_null_0:
        df_train=df_train.fillna({f:0})
        
    return df_train

#####################数据预处理###########################
def data_preprocessing(train,test):

    #######################train中的数据预处理##################
    ###train中的A6
    train_A6 = train['A6']
    for i in range(len(train_A6)):
        train_A6[i] = round(train_A6[i])
    
    train['A6'] = train_A6

    ###train中的A12
    train_A12 = train['A12']
    for i in range(len(train_A12)):
        if train_A12[i] == 98:
            train_A12[i] = 100
        elif train_A12[i] == 107:
            train_A12[i] = 106

    train['A12'] = train_A12

    ###train中的A15
    train_A15 = train['A15']
    for i in range(len(train_A15)):
        train_A15[i] = round(train_A15[i])
            
    train['A15'] = train_A15

    ###train中的A17
    train_A17 = train['A17']
    train_A17_zong = train['A17'].mode().iloc[0]    
    for i in range(len(train_A17)):
        train_A17[i] = round(train_A17[i])
        if train_A17[i] == 89:
            train_A17[i] = train_A17_zong
        if train_A17[i] == 100:
            train_A17[i] = train_A17_zong
        
    train['A17'] = train_A17

    ###train中的A22
    train_A22 = train['A22']
    train_A22_zong = train['A22'].mode().iloc[0]
    for i in range(len(train_A22)):
        if train_A22[i] == 3.5:
            train_A22[i] = train_A22_zong
        #if train_A22[i] == 8:
            #train_A22[i] = train_A22_zong
            
    train['A22'] = train_A22
 
    ###train中的A25
    train_A25_zong = train['A25'].mode().iloc[0]
    A25 = train['A25'].tolist()
    for t in A25:
        if str(t).find("1900/3") != -1:
           A25[A25.index(t)] = train_A25_zong
    train['A25'] = A25
    
    ###train中的A25
    train_A25 = train['A25']
    train_A25_zong = train['A25'].mode().iloc[0]
    for i in range(len(train_A25)):
        if train_A25[i] == 50:
            train_A25[i] = train_A25_zong
        if train_A25[i] == np.nan:
            train_A25[i] = train_A25_zong
            
    train['A25'] = train_A25    
    
    ###train中的B1
    train_B1 = train['B1']
    train_B1_zong = train['B1'].mode().iloc[0]
    for i in range(len(train_B1)):
        if train_B1[i] == 3.5:
            train_B1[i] = train_B1_zong
        elif train_B1[i] == 318:
            train_B1[i] = 320
        elif train_B1[i] == 316:
            train_B1[i] = 320
    
    train['B1'] = train_B1
    
    ###train中的B2
    train_B2 = train['B2']
    for i in range(len(train_B2)):
        if train_B2[i] == 3.6:
            train_B2[i] = 3.5
    
    train['B2'] = train_B2
    
    ###train中的B8
    train_B8 = train['B8']
    train_B8_zong = train['B8'].mode().iloc[0]
    for i in range(len(train_B8)):
        if train_B8[i] == np.nan:
            train_B8[i] = train_B8_zong

    train['B8'] = train_B8        
    
    ###train中的B12
    train_B12 = train['B12']
    train_B12_zong = train['B12'].mode().iloc[0]
    for i in range(len(train_B12)):
        if train_B12[i] == np.nan:
            train_B12[i] = train_B12_zong

    train['B12'] = train_B12    
    
    ###train中的B14
    train_B14 = train['B14']
    train_B14_zong = train['B14'].mode().iloc[0]
    for j in range(len(train_B14)):
        if train_B14[j] ==40:
            train_B14[j] = train_B14_zong
        if train_B14[j] ==256:
            train_B14[j] = 260
        if train_B14[j] ==418:
            train_B14[j] = 420
        if train_B14[j] ==387:
            train_B14[j] = 390
            
    train['B14'] = train_B14

    train = train[train['收率']>= min_threshold]
    train = train[train['B14']>=350]
    train = train[train['B14']<=460]

    #################test中的数据预处理###########################

    test_cols = ['A1','A2','A3','A4','A6','A8','A10','A12','A15','A17','A19','A20','A21','A22','A23','A24','A25','A27','B1','B6','B8','B12','B14']
    
    for f in test_cols:
        test_f = test[f]
        train_f_kinds = train[f].value_counts().index
        try:        ####改了的这里
            test_f_zong = test[f].mode().iloc[0]
        except:
            test_f_zong=np.nan
        for i in range(len(test_f)):
            try:
                if round(test_f[i]) in train_f_kinds:
                    test_f[i] = round(test_f[i])
                else:
                    test_f[i] = test_f_zong
            except:
                test_f[i] = test_f_zong
        test[f] = test_f
 

    return train,test
  
##################time_model################################

class time_model:
    #####################数据预处理###########################
    def data_preprocessing(train,test):
    
            #######################train中的数据预处理##################
        ###train中的A6
        train_A6 = train['A6']
        for i in range(len(train_A6)):
            train_A6[i] = round(train_A6[i])
        
        train['A6'] = train_A6

        ###train中的A12
        train_A12 = train['A12']
        for i in range(len(train_A12)):
            if train_A12[i] == 98:
                train_A12[i] = 100
            elif train_A12[i] == 107:
                train_A12[i] = 106
    
        train['A12'] = train_A12
    
        ###train中的A15
        train_A15 = train['A15']
        for i in range(len(train_A15)):
            train_A15[i] = round(train_A15[i])
                
        train['A15'] = train_A15
     
        ###train中的A17
        train_A17 = train['A17']
        train_A17_zong = train['A17'].mode().iloc[0]    
        for i in range(len(train_A17)):
            train_A17[i] = round(train_A17[i])
            if train_A17[i] == 89:
                train_A17[i] = train_A17_zong
            if train_A17[i] == 100:
                train_A17[i] = train_A17_zong
            
        train['A17'] = train_A17
     
        ###train中的A22
        train_A22 = train['A22']
        train_A22_zong = train['A22'].mode().iloc[0]
        for i in range(len(train_A22)):
            if train_A22[i] == 3.5:
                train_A22[i] = train_A22_zong
            #if train_A22[i] == 8:
                #train_A22[i] = train_A22_zong
                
        train['A22'] = train_A22
     
        ###train中的A25
        train_A25_zong = train['A25'].mode().iloc[0]
        A25 = train['A25'].tolist()
        for t in A25:
            if str(t).find("1900/3") != -1:
               A25[A25.index(t)] = train_A25_zong
        train['A25'] = A25
        
        ###train中的A25
        train_A25 = train['A25']
        train_A25_zong = train['A25'].mode().iloc[0]
        for i in range(len(train_A25)):
            if train_A25[i] == 50:
                train_A25[i] = train_A25_zong
            if train_A25[i] == np.nan:
                train_A25[i] = train_A25_zong
                
        train['A25'] = train_A25    
    
        ###train中的B1
        train_B1 = train['B1']
        train_B1_zong = train['B1'].mode().iloc[0]
        for i in range(len(train_B1)):
            if train_B1[i] == 3.5:
                train_B1[i] = train_B1_zong
            elif train_B1[i] == 318:
                train_B1[i] = 320
            elif train_B1[i] == 316:
                train_B1[i] = 320
        
        train['B1'] = train_B1
        
        ###train中的B2
        train_B2 = train['B2']
        for i in range(len(train_B2)):
            if train_B2[i] == 3.6:
                train_B2[i] = 3.5
        
        train['B2'] = train_B2
        
        ###train中的B8
        train_B8 = train['B8']
        train_B8_zong = train['B8'].mode().iloc[0]
        for i in range(len(train_B8)):
            if train_B8[i] == np.nan:
                train_B8[i] = train_B8_zong
    
        train['B8'] = train_B8        
 
        ###train中的B12
        train_B12 = train['B12']
        train_B12_zong = train['B12'].mode().iloc[0]
        for i in range(len(train_B12)):
            if train_B12[i] == np.nan:
                train_B12[i] = train_B12_zong
    
        train['B12'] = train_B12    
        
        ###train中的B14
        train_B14 = train['B14']
        train_B14_zong = train['B14'].mode().iloc[0]
        for j in range(len(train_B14)):
            if train_B14[j] ==40:
                train_B14[j] = train_B14_zong
            if train_B14[j] ==256:
                train_B14[j] = 260
            if train_B14[j] ==418:
                train_B14[j] = 420
            if train_B14[j] ==387:
                train_B14[j] = 390
                
        train['B14'] = train_B14
    
        
        #######################train中的数据预处理##################
        train = train[train['B14'] >= 350]
        train = train[train['B14'] <= 460]
        train = train[train['收率'] >= 0.85]
        train = train[train['收率'] <= 1]

        ##时间异常值处理
        A25 = train['A25'].tolist()
        for t in A25:
            if str(t).find("1900/3") != -1:
               A25[A25.index(t)] = np.nan
        train['A25'] = A25     
        A5 = train['A5'].tolist()
        for t in A5:
            if str(t).find("1900/1/21 0:00") != -1:
               A5[A5.index(t)] = '21:00:00'
            elif str(t).find("1900/1/29 0:00") != -1:
                A5[A5.index(t)] = '14:00:00'
        train['A5'] = A5
        A9 = train['A9'].tolist()
        for t in A9:
            if str(t).find("1900/1") != -1:
               A9[A9.index(t)] = '23:00:00'
            elif str(t).find('700') != -1:
                A9[A9.index(t)] = '6:30:00'
        train['A9'] = A9
        A11 = train['A11'].tolist()
        for t in A11:
            if str(t).find("1900/1") != -1:
               A11[A11.index(t)] = '21:30:00'
        train['A11'] = A11
        A16 = train['A16'].tolist()
        for t in A16:
            if str(t).find("1900/1") != -1:
               A16[A16.index(t)] = '12:00:00'
        train['A16'] = A16
        A26 = train['A26'].tolist()
        for t in A26:
            if str(t).find("1900/3") != -1:
               A26[A26.index(t)] = '13:00:00'
        train['A26'] = A26
        A55 = test['A5'].tolist()
        for t in A55:
            if str(t).find("1900/1") != -1:
               A55[A55.index(t)] = '22:00:00'
        test['A5'] = A55
        #缺失值的填充
        train['A21']=train['A21'].fillna(50)
        train['B12']=train['B12'].fillna(1200)
        train['B8'] = train['B8'].fillna(45.0)
        train['B5'] = train['B5'].fillna('14:00:00')
        train['B2'] = train['B2'].fillna(3.5)
        train['A24']= train['A24'].fillna('3:00:00')
        train['A24']= train['A24'].fillna('3:00:00')
        #test['A21'] = test['A21'].fillna(50)
        
        data = pd.concat([train,test],axis=0,ignore_index=True)
        train = data[:train.shape[0]]
        test  = data[train.shape[0]:]
    
    
        #################test中的数据预处理###########################
    
        test_cols = ['A1','A2','A3','A4','A6','A8','A10','A12','A15','A17','A19','A20','A21','A22','A23','A24','A25','A27','B1','B6','B8','B12','B14']
        
        for f in test_cols:
            test_f = test[f]
            train_f_kinds = train[f].value_counts().index
            try:        ####改了的这里
                test_f_zong = test[f].mode().iloc[0]
            except:
                test_f_zong=np.nan
            for i in range(len(test_f)):
                try:
                    if round(test_f[i]) in train_f_kinds:
                        test_f[i] = round(test_f[i])
                    else:
                        test_f[i] = test_f_zong
                except:
                    test_f[i] = test_f_zong
            test[f] = test_f
            
            
        test = test[test['收率']>=min_threshold]
        test = test[test['收率']<=max_threshold]
        test = test[test['B14']>=350]
        test = test[test['B14']<=460]
    
    
    
        return train,test
        
    #####################数据特征工程###########################
    def feature_engineering(train,test):
        
            
        target = train.pop('收率')
        data = pd.concat([train,test],axis=0,ignore_index=True)
        ###################特征工程#####################
        
        for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
            try:
                data[f] = data[f].apply(timeTranSecond)
            except:
                print(f,'应该在前面被删除了！')
                
        for f in ['A20','A28','B4','B9','B10','B11']:
            data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
            
        data.drop('样本id',axis = 1, inplace=True)
        data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
        #data['tem_sum'] = data['A6']+data['A10']+data['A12']+data['A15']+data['A17']+data['A21']+data['A27']+data['B6']+data['B8']
        del data['A1']
        #del data['A2']
        del data['A3']
        del data['A4']
        del data['A5']
        del data['B3']
        del data['A13']
        del data['B13']
        #del data['A16']
        data = data.fillna(-1)
        train = data[:train.shape[0]]
        test  = data[train.shape[0]:]
                
                
        X_train = train.values
        X_test = test.values
        y_train = target.values
            
    
       
        return X_train,X_test,y_train
    
    
    
    def model_predict(X_train,X_test,y_train):
        
        param = {'num_leaves': 120,
             'min_data_in_leaf': 30, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_lgb = np.zeros(len(y_train))
        predictions_lgb = np.zeros(len(X_test))
        
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
            oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
            
            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        
        print("lgb score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))
        
        ##### xgb
        
        xgb_params = {'eta': 0.004, 'max_depth': 9, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
        
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_xgb = np.zeros(len(y_train))
        #predictions_xgb = np.zeros(len(y_train_test))
        predictions_xgb = np.zeros(len(X_test))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
            oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
            
            
        print("xgb score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train) / 2))
        
        
        ###rf
        from sklearn.ensemble import RandomForestRegressor
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_rf = np.zeros(len(y_train))
        predictions_rf = np.zeros(len(X_test))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold_rf n°{}".format(fold_+1))
            clf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0)
            clf.fit(X_train[trn_idx], y_train[trn_idx])
            oof_rf[val_idx] = clf.predict(X_train[val_idx])
            predictions_rf += clf.predict(X_test) / folds.n_splits
        print("rf score:{:<8.8f}".format(mean_squared_error(oof_rf, y_train)/2))
        
        
        # 将lgb和xgb的结果进行stacking
        train_stack = np.vstack([oof_lgb,oof_xgb,oof_rf]).transpose()
        test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_rf]).transpose()
        
        folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
        oof_stack = np.zeros(train_stack.shape[0])
        predictions = np.zeros(test_stack.shape[0])
        
        for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
            print("fold {}".format(fold_))
            trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
            val_data, val_y = train_stack[val_idx], y_train[val_idx]
            
            clf_3 = BayesianRidge()
            clf_3.fit(trn_data, trn_y)
            
            oof_stack[val_idx] = clf_3.predict(val_data)
            predictions += clf_3.predict(test_stack) / 10
            
        
        print("time stacking score: {:<8.8f}".format(mean_squared_error(y_train, oof_stack)/2))
        
        print('###############################')
        return predictions,oof_stack
    
    


##################period_model################################
class period_model:
      
    ######################特征工程##########################    
    def feature_engineering(train,test):
    
        
        target = train['收率']
        del train['收率']
        data = pd.concat([train,test],axis=0,ignore_index=True)
        
        
        data=copy_time(data)###将A5复制给A7，并将A8中缺失值赋为0
        for f in ['A20','A28','B4','B9','B10','B11']:
            f_index = get_columns(f,data)  ###得到f在df_train 中对应列号
            data =   Duration_split(f,f_index,data)##将时间段分开为前后两列
        
        for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
            data[f] = data[f].apply(t2s)   ##时间变为分钟
            
        
        data = interval_missing(data)  ###后时间减去前面的时间，并处理缺失值
        
        data = data.drop(['A5'],axis=1)   ###删除开始时间列
        
        
        
        # 删除某一类别占比超过90%的列
        good_cols = list(data.columns)
        for col in data.columns:
            rate = data[col].value_counts(normalize=True, dropna=False).values[0]
            if rate > 0.90:
                good_cols.remove(col)
                print(col,rate)
        
        
        
        good_cols.append('A1')
        good_cols.append('A3')
        good_cols.append('A4')
        good_cols.append('A21')
        
        
        data = data[good_cols]
        
        
        data = data.drop(['样本id'],axis=1)
        
      
        # 有风的冬老哥，在群里无意爆出来的特征
        data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
        
        
        data['tem_sum'] = data['A6']+data['A10']+data['A12']+data['A15']+data['A17']+data['A27']+data['B6']+data['B8']
        
        
        del data['A1']
        del data['A3']
        del data['A4']
        del data['A21']
        
        
        
        train = data[:train.shape[0]]
        test  = data[train.shape[0]:]
        
        
        X_train = train.values
        X_test = test.values
        y_train = target.values
    

        return X_train,X_test,y_train
    
    
    
    ######################训练#####################
    def model_predict(X_train,X_test,y_train):
        
        ###lgb
        param = {'num_leaves': 120,
                 'min_data_in_leaf': 30, 
                 'objective':'regression',
                 'max_depth': -1,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9 ,
                 "bagging_seed": 11,
                 "metric": 'mse',
                 "lambda_l1": 0.1,
                 "verbosity": -1}
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_lgb = np.zeros(len(y_train))
        predictions_lgb = np.zeros(len(X_test))
        
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
            oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
            
            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        
        print("lgb score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)/2))
        
        
        ##### xgb
        xgb_params = {'eta': 0.004, 'max_depth': 9, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
        
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_xgb = np.zeros(len(y_train))
        #predictions_xgb = np.zeros(len(y_train_test))
        predictions_xgb = np.zeros(len(X_test))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
            oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
            
        
        print("xgb score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train) / 2))
        
        
         ###rf
        from sklearn.ensemble import RandomForestRegressor
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_rf = np.zeros(len(y_train))
        predictions_rf = np.zeros(len(X_test))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold_rf n°{}".format(fold_+1))
            clf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0)
            clf.fit(X_train[trn_idx], y_train[trn_idx])
            oof_rf[val_idx] = clf.predict(X_train[val_idx])
            predictions_rf += clf.predict(X_test) / folds.n_splits
        print("rf score:{:<8.8f}".format(mean_squared_error(oof_rf, y_train)/2))
        
        
        
        # 将lgb和xgb的结果进行stacking
        train_stack = np.vstack([oof_lgb,oof_xgb,oof_rf]).transpose()
        test_stack = np.vstack([predictions_lgb, predictions_xgb,predictions_rf]).transpose()
        
        folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
        oof_stack = np.zeros(train_stack.shape[0])
        predictions = np.zeros(test_stack.shape[0])
        
        for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
            print("fold {}".format(fold_))
            trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
            val_data, val_y = train_stack[val_idx], y_train[val_idx]
            
            clf_3 = BayesianRidge()
            clf_3.fit(trn_data, trn_y)
            
            oof_stack[val_idx] = clf_3.predict(val_data)
            predictions += clf_3.predict(test_stack) / 10
            
        
        print("period stacking score: {:<8.8f}".format(mean_squared_error(y_train, oof_stack)/2))
        
        return predictions,oof_stack







def time_period_model_predict(train,test):

    
    train_sr,test_sr = data_preprocessing(train,test)
    
                                    
    train_time = train_sr.copy()
    test_time = test_sr.copy()  
    
    
    X_train,X_test,y_train = time_model.feature_engineering(train_time,test_time)
    
    
    predictions_time,oof_stack_time = time_model.model_predict(X_train,X_test,y_train)
    
    
    
    
    #train_sr,test_sr = data_preprocessing(train,test)
    train_period = train_sr.copy()
    test_period = test_sr.copy()  
    
    
    
    X_train,X_test,y_train=period_model.feature_engineering(train_period,test_period)
    
    
    predictions_period,oof_stack_period=period_model.model_predict(X_train,X_test,y_train)
    
    
    
    
    y_train= train_sr['收率'].values
    # 将时间点time和时间段period的结果进行stacking
    train_stack = np.vstack([np.round(oof_stack_time,3),np.round(oof_stack_period,3)]).transpose()
    test_stack = np.vstack([np.round(predictions_time,3), np.round(predictions_period,3)]).transpose()
    
    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]
        
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
        
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
        
    
    print("time_period_stacking score: {:<8.8f}".format(mean_squared_error(y_train, oof_stack)/2))

    return predictions



def save_predictions(predictions,test_path,save_test_path):
    
    
    df_test = pd.read_csv(test_path,  encoding = 'gb18030', header=None)
    df_test = df_test.drop(0,axis=0)
    sub_df = pd.DataFrame()
    sub_df[0] = df_test[0]
    sub_df[1] = predictions
    sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
    sub_df.to_csv(save_test_path, index=False, header=None)


#######预测最优收率
print('预测最优收率')
optimize_predictions = time_period_model_predict(train,test_optimize)
optimize=pd.DataFrame()
optimize[0] = test_optimize['样本id']
optimize[1] = optimize_predictions
optimize.to_csv('./submit_optimize.csv',index=False, header=None)



######预测FuSai测试集
print('预测FuSai测试集')
FuSai_predictions = time_period_model_predict(train,test)
save_test_path = './submit_FuSai.csv'
save_predictions(FuSai_predictions,test_path,save_test_path)




