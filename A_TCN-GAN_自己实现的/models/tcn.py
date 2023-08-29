import torch
import torch.nn as nn
from torch import squeeze, unsqueeze
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        # 这个是可以看成是成员变量
        # 可以在成员对象上看见这么一个属性是什么
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
        # forward函数里面可以定义大量的自定义的计算


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                 stride=stride, padding=padding, dilation=dilation))
        # 經過conv1，輸出的size其實是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)
        # 这个说到底不就是一维的卷积的操作吗，不就是可以作为特征的提取器吗
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨脹係數：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 確定每一層的輸入通道數
            out_channels = num_channels[i]  # 確定每一層的輸出通道數
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
# 这个是可以进行随机的组合的

class TCN(nn.Module):
    def __init__(self,num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCN,self).__init__()
        self.tcn=TemporalConvNet(
            num_inputs, num_channels, kernel_size=3, dropout=0.2)
        self.dropout = nn.Dropout(dropout)
        self.liner = nn.Linear(num_channels[-1],num_inputs) # 线性层的数量是可以进行选择的
    def forward(self, x,channel_last=True):
        y1 = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y1.transpose(1, 2))  # 在这条语句里面设三个类都有体现