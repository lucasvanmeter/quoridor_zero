import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResBlock(nn.Module):
    def __init__(self, nchan):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(nchan, nchan, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nchan),
            nn.ReLU(),
            nn.Conv2d(nchan, nchan, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nchan)
        )
        
    def forward(self, x): return F.relu(x + self.convs(x))
            
def conv_block(ni, nf, kernel = 3):
    return nn.Sequential(
            nn.Conv2d(ni, nf, kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU()
    )

class QuoridorNNet(nn.Module):
    """
    Input:
    Description: 
    Output: 
    """
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = 9, 9
        self.action_size = 64+64+81
        self.args = args

        super(QuoridorNNet, self).__init__()
        self.stem = conv_block(1, args.num_channels) 
        self.resblock1 = ResBlock(args.num_channels)
        self.resblock2 = ResBlock(args.num_channels)
        self.resblock3 = ResBlock(args.num_channels)
        self.resblock4 = ResBlock(args.num_channels)
        
        self.downsample = conv_block(args.num_channels, 128, 1)

        self.fc1 = nn.Linear(2+args.num_channels*(self.board_x)*(self.board_y), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.policy = nn.Linear(512, self.action_size)
        self.value = nn.Linear(512, 1)

    def forward(self, src):
        """
        Arguments:
            src: Tensor, shape [batch_size, 2+81]

        Returns:
            output Tensor of shape ````
        """
        sRemWalls = src[:,:2]                              # batch_size x 2
        sBoards = src[:,2: 2 + 81]                         # batch_size x 81
        s = sBoards.view(-1,9,9)                           # batch_size x board_x x board_y
        s = s.view(-1,1, self.board_x, self.board_y)       # batch_size x 1 x board_x x board_y
        s = self.stem(s)                                   # batch_size x num_channels x board_x x board_y
        s = self.resblock1(s)                              # batch_size x num_channels x board_x x board_y
        s = self.resblock2(s)                              # batch_size x num_channels x board_x x board_y  
        s = self.resblock3(s)                              # batch_size x num_channels x board_x x board_y
        s = self.resblock4(s)                              # batch_size x num_channels x board_x x board_y 
        
        s = s.view(-1, self.args.num_channels*(self.board_x)*(self.board_y)) # batch_size x num_channels*board_x*board_y
        s = torch.cat((sRemWalls, s), dim=1)                               # add on remaining wall info
        s = self.fc1(s)
        s = F.dropout(F.relu(self.fc_bn1(s)), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.policy(s)                                                                         # batch_size x action_size
        v = self.value(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
