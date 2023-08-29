import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm


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
        self.linear = nn.Linear(num_channels[-1], output_size)
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
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
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
        self.tcn = TCN(noise_size, output_size, num_channels, kernel_size, dropout=0.2)

    def forward(self, x, channel_last=True):
        return torch.tanh(self.tcn(x, channel_last))



def main():
    noise=torch.randn(100,1,46)
    data=torch.randn(100,1,45)
    noise_size=46
    output_size=45
    n_layers=3
    n_channel=2
    kernel_size=2
    dropout=0.2
    input_size=44
    gen=Generator(noise_size,output_size,n_layers,n_channel,kernel_size,dropout)
    c=gen(noise)
    print(c.shape)
    dis=Discriminator(input_size=45, n_layers=7, n_channel=9, kernel_size=2, dropout=0.2)
    # netG = Generator(noise_size=3, output_size=4, n_layers=8, n_channel=7, kernel_size=2, dropout=0.2)
    d=dis(data) # 输出的值就是用来判断真假的
    a=d.mean().item()
    # 这个是一个是tensor类型的数值一个是float类型的数值
    print(d)
    print(d.shape)
    print(a) # 这个是一次的loss的值


    # print(gen)
if __name__=='__main__':
    main()

# netG = Generator(noise_size=3, output_size=4, n_layers=8, n_channel=7, kernel_size=2, dropout=0.2)
# netD = Discriminator(input_size=4, n_layers=7, n_channel=9, kernel_size=2, dropout=0.2).to(device)



