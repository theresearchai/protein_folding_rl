'''
Dual head residual network used for AlphaGo Zero, also parts of a RNN model (incomplete)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

from collections import OrderedDict

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[128], deepths=[20], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),   
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, inplanes, outplanes, grid_len = 201):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.inplanes = inplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(grid_len * grid_len, outplanes)
        

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        probas = self.logsoftmax(x).exp()

        return probas

class ValueNet(nn.Module):
    def __init__(self, inplanes, outplanes, grid_len = 201):
        super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(grid_len * grid_len, 256)
        self.fc2 = nn.Linear(256, 1)
        

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        winning = torch.tanh(self.fc2(x))
        return winning

class DualRes(nn.Module):
    
    def __init__(self, in_channels, n_classes, grid_len = 201, cuda = False, tpu = False, dev = None, blocks_size = [128], deepths = [20], *args, **kwargs):
        super().__init__()
        self.use_cuda = cuda
        self.use_tpu = tpu
        self.dev = dev
        self.in_channels = in_channels
        self.grid_len = grid_len
        self.encoder = ResNetEncoder(in_channels, blocks_size = blocks_size, deepths = deepths, *args, **kwargs)
        self.policy = PolicyNet(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes, grid_len)
        self.value = ValueNet(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes, grid_len)
        
        
    def forward(self, x):
        x = self.encoder(x)
        pi = self.policy(x)
        v = self.value(x)
        return pi, v
    
    def predict(self, x):
        if self.use_tpu:
            x = torch.from_numpy(x).to(self.dev)
        else:
            x = torch.FloatTensor(x.astype(np.float64))
        if self.use_cuda:
            x = x.contiguous().cuda()
        x = x.view(1, self.in_channels, self.grid_len, self.grid_len)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(x)
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
class RNNEncoder1(nn.Module):
    def __init__(self, in_channels, blocks_size = [128], deepths = [1], *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, blocks_size = blocks_size, deepths = deepths, *args, **kwargs)
        self.conv = nn.Conv2d(self.encoder.blocks[-1].blocks[-1].expanded_channels, 1, kernel_size = 1)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return x
    
    
class RNNEncoder2(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, hidden_size, kernel_size = 3), nn.BatchNorm2d(1), nn.ReLU())
        
    def forward(self, x):
        x = self.layers(x)
        return x
        
    
class RNN(nn.Module):
    def __init__(self, in_channels, n_classes, grid_len, hidden_size, cuda = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_cuda = cuda
        self.enc = RNNEncoder1(in_channels, blocks_size = [hidden_size])
        
        self.i2h = nn.Linear(grid_len * grid_len + hidden_size, hidden_size)
        self.i2p = nn.Linear(grid_len * grid_len + hidden_size, n_classes)
        self.i2v = nn.Linear(grid_len * grid_len + hidden_size, 1)
        self.logSoftmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, x, hidden):
        x = self.enc(x)
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        p = self.i2p(combined)
        p = self.logSoftmax(p).exp()
        v = self.i2v(combined)
        return p, v, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def predict(self, x, hidden):
        x = torch.FloatTensor(x.astype(np.float64))
        if self.use_cuda:
            x = x.contiguous().cuda()
        x = np.expand_dim(x, 0)
        self.eval()
        with torch.no_grad():
            pi, v, hidden = self.forward(x)
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
class GRUNet(nn.Module):
    def __init__(self, in_channels, n_classes, hidden_size, drop_prob = 0.2, cuda = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_cuda = cuda
        
        self.gru = nn.GRU(in_channels, hidden_size, batch_first = True)
        self.i2p = nn.Linear(hidden_size, n_classes)
        self.i2v = nn.Linear(hidden_size, 1)
        self.logSoftmax = nn.LogSoftmax(dim = 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        x = torch.flatten(x, start_dim = 2)
        x = torch.transpose(x, 1, 2)
        out, h = self.gru(x, h)
        p = self.i2p(self.relu(out[:, -1]))
        p = self.logSoftmax(p).exp()
        v = self.i2v(self.relu(out[:, -1]))
        return p, v, h
        
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    def predict(self, x, hidden):
        x = torch.FloatTensor(x.astype(np.float64))
        if self.use_cuda:
            x = x.contiguous().cuda()
        x = np.expand_dim(x, 0)
        self.eval()
        with torch.no_grad():
            pi, v, hidden = self.forward(x)
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]