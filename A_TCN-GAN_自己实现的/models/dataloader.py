import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Path of the generator checkpoint')
parser.add_argument('--output_path', required=True, help='Path of the output .npy file')
# 增量的路径
parser.add_argument('--delta_path', default='', help='Path of the file containing the list of deltas for conditional '
                                                     'generation')
parser.add_argument('--dataset_path', required=True, help="Path of the dataset for normalization")
parser.add_argument('--nUser', default=1, help='Number of the users')
parser.add_argument('--nTime', default=1, help='Length of the times (hour)')
parser.add_argument('--nSim', default=1, help='Number of the simulations')
parser.add_argument('--mType', default='lstm', help='Generation model type')

args = parser.parse_args()

data=pd.read_csv('2.csv')
features_process = data.iloc[:,2:]
df_features = features_process
df_features1 = df_features[['温度']]
df_features2 = df_features[['湿度']]
df_features3 = df_features[['光照强度']]
df_features4 = df_features[['电压']]
df_features
from sklearn.preprocessing import MinMaxScaler
df_features1 = df_features1.fillna(method = 'ffill')
df_features2 = df_features2.fillna(method = 'ffill')
df_features3 = df_features3.fillna(method = 'ffill')
df_features4 = df_features4.fillna(method = 'ffill')
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler1 = MinMaxScaler(feature_range=(-1, 1))


date = list(data['日期'])
date_form = []

for i in date:
    i = i.replace('/','-')
    date_form.append(i)

dates = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in date_form]

df_features['温度'] = scaler1.fit_transform(df_features['温度'].values.reshape(-1,1))
scaler_2 = MinMaxScaler(feature_range=(-1, 1))
df_features['湿度'] = scaler_2.fit_transform(df_features['湿度'].values.reshape(-1,1))
scaler_3 = MinMaxScaler(feature_range=(-1, 1))# 这个是进行规整后的形状是怎么样的
df_features['光照强度'] = scaler_3.fit_transform(df_features['光照强度'].values.reshape(-1,1))
scaler_4 = MinMaxScaler(feature_range=(-1, 1))
df_features['电压'] = scaler_4.fit_transform(df_features['电压'].values.reshape(-1,1))



look_back_windows = 17 #
# 这个是可以看做是滑动窗口的大小是怎么样的
batch_size=13

def load_data(stock, n_timestamp, true, dates):  # timetamp的意思是时间戳，这个里面指的就是滑动窗口的大小为多少

    # load_data里面的数据指代不明确呀df_features, look_back_windows, df_features1, dates

    # 在送入到神经网络里面的时候是不是进行归一化的操作集可以了呢？这个是不可以的还必须进行数据的格式的装换
    data_raw = stock.values  # 这个是传入的数据，然后将特征值给取出来
    # convert to numpy array
    data = []

    # 不管是什么样的训练数据是必须进行装换的
    #  dataset["train"] = torch.from_numpy(np.array(train_data)).float()这个是将原始的数据转换成tensor的格式

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - n_timestamp):  # 这个是创建出所有可能的时间序列的长度是怎么样的
        data.append(data_raw[index: index + n_timestamp])

    data = np.array(data);  # 这个是将原来所有的数据转换成array的格式来进行处理,
    # 此时data的数量不是有着成千上百种的可能性
    # data现在就是（数据长度、滑动窗口大小、数据特征的维度）这个样子的三维数据长度

    # 这个数据集的划分是20%的测试集，80%的训练集
    test_set_size = int(np.round(0.2 * data.shape[0]));  # 这个输入的数据的长度这么是这个样子的，这个是20%作为验证集
    train_set_size = data.shape[0] - (test_set_size);  # 这个是所有的数据都在这里面将所有的数据都拿出来了

    # test_size和trian_size加起来是整个数据的长度
    train_set_size_list = train_set_size + n_timestamp

    # 这个是分为两波数据来进行测试的
    # 这个就不是说将数据直接是归一化后直接送入到网络里面进行处理
    train_dates = dates[:train_set_size]
    test_dates = dates[train_set_size_list:]
    train_true = true[:train_set_size]
    test_true = true[train_set_size_list:]
    # 这个具体的时间的信息也只是在画图的时候用到了
    # 现在知道为什么输入的时候，输入不进去了，这个是将object对象也给考虑进去了

    x_all = data[:, :-1, :]  # 这个样子的形状是三维数组的表现形式是怎么样的
    # 这个数据是7个结构的特征是怎么样的
    # 原始数据是三维的才可以进行三维的变换

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_all, x_train, y_train, x_test, y_test, data, train_true, test_true, train_dates, test_dates]


# 这个就是自己来制作滑动窗口
#
x_all, x_train, y_train, x_test, y_test, data, train_true, test_true, train_dates, test_dates = load_data(df_features,
                                                                                                          look_back_windows,
                                                                                                          df_features1,
                                                                                                          dates)



x_all, x_train, x_valid, y_train, y_valid = np.array(x_all), np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
x_all, \
x_train, x_valid, y_train, y_valid = torch.tensor(x_all, dtype = torch.float), torch.tensor(x_train, dtype = torch.float), torch.tensor(x_valid, dtype = torch.float), torch.tensor(y_train, dtype = torch.float), torch.tensor(y_valid, dtype = torch.float)
# print(x_train.shape)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True, drop_last=True)
