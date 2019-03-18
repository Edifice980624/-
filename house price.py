#import modules
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
#we will use regular expression for passenger names
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
#import data
train = pd.read_csv('D:/train.csv')
test = pd.read_csv('D:/test.csv')
#we put both data frames in a list to modify all data easily


#data quick check
print(train.head())
print(train.info())
print('\n', test.head())
print(test.info())
train.describe()

#数据的初步处理
#drop = ['时间']
#train.drop(drop, axis=1, inplace=True)
#test.drop(drop, axis=1, inplace=True)

#train.loc[ (train.地铁线路.notnull()), '地铁线路' ] = 1
#train.loc[ (train.地铁线路.isnull()), '地铁线路' ] = 0
train.loc[ (train.装修情况.isnull()), '装修情况' ] = 0
train.loc[ (train.居住状态.isnull()), '居住状态' ] = 0
#train.loc[ (train.出租方式.isnull()), '出租方式' ] = 0
#test.loc[ (test.地铁线路.notnull()), '地铁线路' ] = 1
#test.loc[ (test.地铁线路.isnull()), '地铁线路' ] = 0
test.loc[ (test.装修情况.isnull()), '装修情况' ] = 0
test.loc[ (test.居住状态.isnull()), '居住状态' ] = 0
#test.loc[ (test.出租方式.isnull()), '出租方式' ] = 0


#处理不合适的数据
train = train[train['房屋面积'] !=1]
train = train[train['房屋面积'] <0.145]

train['房屋面积'] = train['房屋面积']*6042
test['房屋面积'] = test['房屋面积']*6042
train['总楼层']=train['总楼层']*55
test['总楼层']=test['总楼层']*55
train['楼层']=((train['楼层']+1)*33/100-0.165)*train['总楼层']
test['楼层']=((test['楼层']+1)*33/100-0.165)*test['总楼层']


train['小区房屋出租数量'] = train['小区房屋出租数量']*256
test['小区房屋出租数量'] = test['小区房屋出租数量']*256
train['距离'] = train['距离']*1200
test['距离'] = test['距离']*1200
train.loc[ (train.距离.isnull()), '距离' ] = 99999
test.loc[ (test.距离.isnull()), '距离' ] = 99999
train['房屋总数量']=train['卧室数量']+train['厅的数量']+train['卫的数量']
test['房屋总数量']=test['卧室数量']+test['厅的数量']+test['卫的数量']
train['平均面积']=train['房屋面积']/train['房屋总数量']
test['平均面积']=test['房屋面积']/test['房屋总数量']

'''
#有无朝南的
cxshuzi = train['房屋朝向'].str.contains('南')
cxshuzi=1*cxshuzi
train['是否朝南']=cxshuzi
cxshuzi = test['房屋朝向'].str.contains('南')
cxshuzi=1*cxshuzi
test['是否朝南']=cxshuzi
'''


#几个朝南的
s = pd.Series(train['房屋朝向'])
s=pd.DataFrame(s.str.count("南"))
s.columns=['朝南数量']
train['朝南数量']=s
s = pd.Series(test['房屋朝向'])
s=pd.DataFrame(s.str.count("南"))
s.columns=['朝南数量']
test['朝南数量']=s
train['朝南面积']=train['朝南数量']/train['房屋总数量']*train['房屋面积']
test['朝南面积']=test['朝南数量']/test['房屋总数量']*test['房屋面积']


'''
s = pd.Series(train['房屋朝向'])
s=pd.DataFrame(s.str.count("北"))
s.columns=['朝北数量']
train['朝北数量']=s
s = pd.Series(test['房屋朝向'])
s=pd.DataFrame(s.str.count("北"))
s.columns=['朝南数量']
test['朝北数量']=s

s = pd.Series(train['房屋朝向'])
s=pd.DataFrame(s.str.count("西"))
s.columns=['朝西数量']
train['朝西数量']=s
s = pd.Series(test['房屋朝向'])
s=pd.DataFrame(s.str.count("西"))
s.columns=['朝西数量']
test['朝西数量']=s

s = pd.Series(train['房屋朝向'])
s=pd.DataFrame(s.str.count("东"))
s.columns=['朝东数量']
train['朝东数量']=s
s = pd.Series(test['房屋朝向'])
s=pd.DataFrame(s.str.count("东"))
s.columns=['朝东数量']
test['朝东数量']=s


train.drop(['房屋朝向'], axis=1, inplace=True)
test.drop(['房屋朝向'], axis=1, inplace=True)
'''


#也许没用，有可能并不是一一对应
'''
xiaoquming=train['小区房屋出租数量'].append(test['小区房屋出租数量'])
le = preprocessing.LabelEncoder()
le.fit(xiaoquming)
le.classes_
train['小区名']=le.transform(train['小区房屋出租数量'])
test['小区名']=le.transform(test['小区房屋出租数量'])
'''




#检查地铁线路对租金的影响
'''
var  = '地铁线路'
data = pd.concat([train['月租金'], train[var]], axis=1)
data.plot.scatter(x=var, y='月租金', ylim=(0, 100))
'''
#画热力图
'''
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
'''

#对朝向的处理


#为防止有新数据，先将train和test的朝向合并
cx=train['房屋朝向'].append(test['房屋朝向'])

le = preprocessing.LabelEncoder()
le.fit(cx)
le.classes_
train['房屋朝向']=le.transform(train['房屋朝向'])
test['房屋朝向']=le.transform(test['房屋朝向'])



#OneHotEncoder处理，但是调参有问题
'''
cxshuzi=le.transform(cx)
cxreshape=cxshuzi.reshape(-1,1)
ote=preprocessing.OneHotEncoder()
ote.fit(cxreshape)
trainreshape=train['房屋朝向'].reshape(-1,1)
trainreshape=ote.transform(trainreshape).toarray()
testreshape=test['房屋朝向'].reshape(-1,1)
testreshape=ote.transform(testreshape).toarray()
train.drop(['房屋朝向'], axis=1, inplace=True)
test.drop(['房屋朝向'], axis=1, inplace=True)

trainadd=pd.DataFrame(trainreshape)
trainnew=train.join(trainadd)
train=trainnew
testadd=pd.DataFrame(testreshape)
testnew=test.join(testadd)
test=testnew

'''







y_train = train.pop('月租金')
# all_data = pd.concat([train, test])
res = pd.DataFrame()
res['id'] = test['id']



X, y, X_test = train, y_train, test

params = {'boosting_type': 'gbdt',
          'num_leaves': 28,
          'max_depth': -1,
          'objective': 'regression',
          'learning_rate': 0.1,
          'seed': 2018,
          'num_threads': -1,
          'max_bin': 425,
          "metric": "rmse",
          # "lambda_l1": 0.1,
          "lambda_l2": 0.2,
          }



x_train = X
x_test = test.drop(columns=["id"])

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_data = lgb.Dataset(x_train, label=y_train.values.flatten())
val_data = lgb.Dataset(x_val, label=y_val.values.flatten())

clf = lgb.train(params, train_set=train_data, num_boost_round=50000, valid_sets=[train_data, val_data],
                  valid_names=['train', 'valid'], early_stopping_rounds=1000, feval=None, verbose_eval=50)





res['price'] = clf.predict(x_test)
print('price mean:%d', res['price'].mean())
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
#res.to_csv("jieguo111111.csv", index=None)


