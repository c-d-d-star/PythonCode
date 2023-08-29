import argparse
import math
import os
from collections import OrderedDict
import datetime

from tensorboardX import writer
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import warnings

from torch.backends import cudnn
from torch.nn import init

warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as Data

from torch.nn.utils import weight_norm

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp1 = Chomp1d(padding)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout2d(dropout)

    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp2 = Chomp1d(padding)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout2d(dropout)

    self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                             self.conv2, self.chomp2, self.relu2, self.dropout2)
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    #self.conv1.weight.data.normal_(0, 0.01)
    nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
    #self.conv2.weight.data.normal_(0, 0.01)
    nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
    if self.downsample is not None:
        #self.downsample.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))

  def forward(self, x):
    net = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(net + res)

class TemporalConvNet(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,max_length=16):
    super(TemporalConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size-1) * dilation_size, dropout=dropout)]

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)  # 经过一层卷积和一层
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, channel_last=True):
        # If channel_last, the expected format is (batch_size, seq_len, features)
        y1 = self.tcn(x.transpose(1, 2) if channel_last else x)

        return self.linear(y1.transpose(1, 2))

class Discriminator(nn.Module):
    """Discriminator using casual dilated convolution, outputs a probability for each time step
    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers，这个和你的计算的方式是一样的
        dropout: (float in [0-1]): dropout rate
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    这个才是应该这么做的啊
    """
    def __init__(self, input_size, n_layers, n_channel, kernel_size, dropout=0):
        super().__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, num_channels, kernel_size, dropout)

    def forward(self, x, channel_last=True):
        return torch.sigmoid(self.tcn(x, channel_last))

class Generator(nn.Module):
    """Generator using casual dilated convolution, expecting a noise vector for each timestep as input
    Args:
        noise_size (int): dimensionality (channels) of the input noise
        output_size (int): dimenstionality (channels) of the output sequence
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, outputsize)
    """

    def __init__(self, noise_size, output_size, n_layers, n_channel, kernel_size, dropout=0.2):
        super().__init__()
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(noise_size, output_size, num_channels, kernel_size, dropout)

    def forward(self, x, channel_last=True):
        return torch.tanh(self.tcn(x, channel_last))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="amazon_dataset", help='dataset to use')
parser.add_argument('--dataset_path',default = "data/amazon_dataset.csv", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--nz', type=int, default=1, help='dimensionality of the latent vector z')
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='Amazon', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=10, help='number of epochs after which saving checkpoints')
parser.add_argument('--dis_type', default='cnn', choices=['cnn','lstm'], help='architecture to be used for discriminator to use')
parser.add_argument('--gen_type', default='lstm', choices=['cnn','lstm'], help='architecture to be used for generator to use')
parser.add_argument('--start_epoch',type=int, default=0, help='number of epochs start to train for')
parser.add_argument('--seq_len',type=int, default=49, help='number of epochs start to train for')

opt = parser.parse_args()

data=pd.read_csv('8000条正常的数据的值.csv')
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

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")
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


look_back_windows =17#
# 这个是可以看做是滑动窗口的大小是怎么样的
batch_size=4

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
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True, drop_last=True)
mse = torch.nn.MSELoss(reduce=True, size_average=True) # 定义的是均方误差
device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = opt.nz
in_dim =  opt.nz


criterion = nn.BCELoss().to(device)

netG = Generator(noise_size=in_dim, output_size=4, n_layers=8, n_channel=7, kernel_size=2, dropout=0.2).to(device)
netD = Discriminator(input_size=4, n_layers=7, n_channel=9, kernel_size=2, dropout=0.2).to(device)




print(netG)
# print(netD)

#Generate fixed noise to be used for visualization
fixed_noise = torch.randn(opt.batchSize, opt.seq_len, nz, device=device)

real_label = 1
fake_label = 0


def denormalize(self, x):
    """Revert [-1,1] normalization"""
    if not hasattr(self, 'max') or not hasattr(self, 'min'):
        raise Exception("You are calling denormalize, but the input was not normalized")
    return 0.5 * (x * self.max - x * self.min + self.max + self.min)

def check_dir(path):
    if not os.path.exists(path):
        # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        #
        os.makedirs(path)


def save_model(path, model_name, model):
    check_dir(path)
    torch.save(model.state_dict(), os.path.join(path, '%s.pth' % model_name))


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

fake_list=[]
cnt = 0

for epoch in range(opt.start_epoch, opt.epochs):
        for i, data in enumerate(train_loader, 0):
            niter = epoch * len(train_loader) + i
        # data = data.unsqueeze(0)

        # Save just first batch of real data for displaying
            if i == 0:
                real_display = data.cpu()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
            correct = 0
        # Train with real data
            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device, dtype=torch.float)
        #print(real.size)
            output = netD(real)
        # print(real.shape)
        # print(output.shape)
            score= output-real
            thread=torch.full((batch_size, seq_len, 4),1.5, device=device, dtype=torch.float)
            # a=score-thread
        # if(abs(a.mean().item())>1.5):
        #       correct+=1
        #       acc = 1. * correct / real.size(1)
            errD_real = criterion(output, label)
        # # 这个值是约接近于1越好
        #
        #       pred = output.max(1, keepdim=True)[1]
        #
        # # acc=1. * correct / real.size(1)


        #print(errD_real.size)
            errD_real.backward()

            D_x = output.mean().item()# 这个是需要比较重要的,这个是越接近于1 是越好的
        #print(D_x)


        # Train with fake data
            noise = torch.randn(batch_size, seq_len, nz, device=device)
        # print(noise.size)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())

            errD_fake = criterion(output, label)
            errD_fake.backward()
        # 这个是越接近于0最好

            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
        # errD.backward()

            optimizerD.step()
        # print(errD)

    #
        # # Visualize discriminator gradients
        # for name, param in netD.named_parameters():
        #     writer = SummaryWriter(log_dir="add_histogram_demo_data/log")
        #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()
#
        # Visualize generator gradients
        for name, param in netG.named_parameters():
            writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        print('[%d/%d][%d/%d] Loss_D: %.7f Loss_G: %.7f D(x) : %.4f D(G(z)): %.7f / %.7f Score:%.4f'% (epoch, opt.epochs, i, len(train_loader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,output.mean().item()), end='\n')
#         # #### Report metrics #####
        #

        #
        writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
        writer.add_scalar('GeneratorLoss', errG.item(), niter)
        writer.add_scalar('D of X', D_x, niter)
        writer.add_scalar('D of G of z', D_G_z1, niter)

    # ##### End of the epoch #####

    # Checkpoint
        if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
            torch.save(netG.state_dict(), f'{opt.outf}/{opt.run_tag}_netG_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'{opt.outf}/{opt.run_tag}_netD_epoch_{epoch}.pth')

        # real = denormalize(real_display)
        # real = torch.flatten(real)
        # real = real.detach().numpy()
        # plt.plot(real)
        #
        # fake = netG(fixed_noise).cpu()
        # fake = train_loader.denormalize(fake)
        # fake = torch.flatten(fake)
        # fake = fake.detach().numpy()
        # plt.plot(fake)
        # 这个就是判断为正的数量一个是判断为负的数量
        # plt.savefig(f'fig/{epoch}epoch_result.png')
        # plt.clf()

# def testmodel(test_loader,generator, discriminator, threshold):
#     cnt=0
#     output=generator(test_loader)
#     if (output-test_loader>threshold):
#         cnt+=1


# torch.save(netG.state_dict(), 'G:\\Model_dict\\netDtrain')
# torch.save(netD.state_dict(), 'G:\\Model_dict\\netGtrain')
# gan的优化是可以从损失上尽心下手的
# def train(discriminator,geneartor,data):
#         discriminator.train()
#         geneartor.train()
#         # data = data.unsqueeze(0)
#         ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         ###########################
#
#         # Train with real data
#         netD.zero_grad()
#         real = data.to(device)
#         batch_size, seq_len = real.size(0), real.size(1)
#         label = torch.full((batch_size, seq_len, 1), real_label, device=device, dtype=torch.float)
#         #print(real.size)
#         output = netD(real)
#         # print(real.shape)
#         # print(output.shape)
#
#
#         # output1=torch.full((batch_size, seq_len, 4),output, device=device, dtype=torch.float)
#         # a=score-thread
#         # if(abs(a.mean().item())>1.5):
#         #       correct+=1
#         #       acc = 1. * correct / real.size(1)
#         errD_real = criterion(output, label)
#         # # 这个值是约接近于1越好
#         #
#         #       pred = output.max(1, keepdim=True)[1]
#         #
#         # # acc=1. * correct / real.size(1)
#         # erDres = criterion(output1, real)
#         # erDres.backward()
#
#         #print(errD_real.size)
#         errD_real.backward()
#         # optimizerD.step()
#
#         D_x = output.mean().item()# 这个是需要比较重要的,这个是越接近于1 是越好的
#         #print(D_x)
#
#
#         # Train with fake data
#         noise = torch.randn(batch_size, seq_len, nz, device=device)
#         # print(noise.size)
#         fake = netG(noise)
#         label.fill_(fake_label)
#         output = netD(fake.detach())
#
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         # 这个是越接近于0最好
#
#         # errD_real.backward()
#         # optimizerD.step()
#         D_G_z1 = output.mean().item()
#         errD = errD_real + errD_fake
#         # errD.backward()
#         # optimizerD.step()
#         # print(errD)
#    #
#         # # Visualize discriminator gradients
#         # for name, param in netD.named_parameters():
#         #     writer = SummaryWriter(log_dir="add_histogram_demo_data/log")
#         #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)
#
#         ############################
#         # (2) Update G network: maximize log(D(G(z)))
#         ###########################
#         netG.zero_grad()
#         label.fill_(real_label)
#         output = netD(fake)
#
#         errG = criterion(output, label)
#         errG.backward()
#         D_G_z2 = output.mean().item()

        # optimizerG.step()

# def test(discriminator,geneartor, data):
#             # data = data.unsqueeze(0)
#             ############################
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ###########################
#
#             # Train with real data
#         discriminator.eval()
#         geneartor.eval()
#         # real = data.to(device)
#         real = data.to(device)
#         batch_size, seq_len = real.size(0), real.size(1)
#         label = torch.full((batch_size, seq_len, 1), real_label, device=device, dtype=torch.float)
#         # print(real.size)
#         output = netD(real)
#         #     # print(real.shape)
#         #     # print(output.shape)
#         score = output - real
#         # print(score.mean().item())
#
#         thread = torch.full((batch_size, seq_len, 4), 1, device=device, dtype=torch.float)
#         a = score - thread
#         print(abs(score.mean().item()))
#         # a /= seq_len
#
#         # if(abs(a.mean().item())>1.5):
#         #     correct+=1
#         #     acc = 1. * correct / real.size(1)
#         errD_real = criterion(output, label)
#             # # 这个值是约接近于1越好
#             #
#             #       pred = output.max(1, keepdim=True)[1]
#             #
#             # # acc=1. * correct / real.size(1)
#
#             # print(errD_real.size)
#
#
#         D_x = output.mean().item()  # 这个是需要比较重要的,这个是越接近于1 是越好的
#             # print(D_x)
#
#             # Train with fake data
#         noise = torch.randn(batch_size, seq_len, nz, device=device)
#             # print(noise.size)
#         fake = netG(noise)
#         label.fill_(fake_label)
#         output = netD(fake.detach())
#
#         errD_fake = criterion(output, label)
#             # 这个是越接近于0最好
#
#         D_G_z1 = output.mean().item()
#
#         errD = errD_real + errD_fake
            # errD.backward()
        # correct=0
        # if((output.argmax(2)>x_train.argmax(2)).type(torch.float).sum().item()):
        #     correct+=1
        #     correct /= x_test.size(1)

        # correct += (output.argmax(1) == errD_fake).type(torch.float).sum().item()
        # print(f"Test Error: \n Accuracy: {(abs(correct)):>0.4f}%\n")

            # print(errD)

            #
            # # Visualize discriminator gradients
            # for name, param in netD.named_parameters():
            #     writer = SummaryWriter(log_dir="add_histogram_demo_data/log")
            #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

        label.fill_(real_label)
        output = netD(fake)

        errG = criterion(output, label)
        D_G_z2 = output.mean().item()



        # Visualize generator gradients
        # for name, param in netG.named_parameters():
        #     writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x) : %.4f D(G(z)): %.4f / %.4f '% (epoch, opt.epochs, i, len(train_loader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,), end='')
        # #### Report metrics #####


#
# for epoch in range(opt.start_epoch, opt.epochs):
#         for i, data in enumerate(train_loader, 0):
#             niter = epoch * len(train_loader) + i
#             #         # data = data.unsqueeze(0)
#             #
#             #         # Save just first batch of real data for displaying
#             if i == 0:
#                 real_display = data.cpu()
#
#             train(netD,netG,x_train)
#             test(netD, netG,x_train)
# def test(model, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for i, data in enumerate(test_loader, 0):
#             x, y = data
#             x = x.cuda()
#             y = y.cuda()
#             optimizer.zero_grad()
#             y_hat = model(x)
#             test_loss += criterion(y_hat, y).item()
#             pred = y_hat.max(1, keepdim=True)[1]
#             correct += pred.eq(y.view_as(pred)).sum().item()
#         test_loss /= (i+1)
#         print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_data), 100. * correct / len(test_data)))

# 现在是我1只要知道correct的值就可以进行查全率，召回率之间的计算了
# 里面的数据我是知道的
# 故障的数量是我设置的




Y = [0.3728,0.4852,0.4083,0.5645,0.4732,0.3265,0.2150,0.2381,0.245,0.2345]

Y1 = [0.96,0.5574,0.6444,0.4785,0.5968,0.9048,0.9331,0.9549,0.94,0.9784]  # 这个就是显示的是每个信息是怎么样的
Y2 = [0.4782,0.5148,0.5895,0.6618,0.6299,0.5848,0.5478,0.5694,0.5605,0.5537]

bar_width = 0.25
tick_label = ['10','20','30','40','50','60','70','80','90','100']