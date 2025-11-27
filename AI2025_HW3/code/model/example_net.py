import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        # self.to(device)
        ########################
        # TODO: your code here #
        board_x, board_y = observation_size

        # 卷积层处理二维棋盘特征
        self.conv1 = nn.Conv2d(1, config.num_channels, kernel_size=3, padding=1)  # 保持尺寸不变
        self.conv2 = nn.Conv2d(config.num_channels, config.num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.num_channels)
        self.bn2 = nn.BatchNorm2d(config.num_channels)

        # 展平后进入全连接层
        conv_out_size = config.num_channels * board_x * board_y
        self.fc1 = nn.Linear(conv_out_size, config.linear_hidden[0])
        self.fc2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])

        # policy head & value head
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.to(device)
        ########################
    
    def forward(self, s: torch.Tensor):
        ########################
        # TODO: your code here #
        s = s.unsqueeze(1)  # 增加通道维度：batch_size x 1 x board_x x board_y
        x = self.relu(self.bn1(self.conv1(s)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平：batch_size x (channels * board_x * board_y)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        pi = self.policy_head(x)
        v = self.value_head(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
        ########################