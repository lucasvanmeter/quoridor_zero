import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def conv_block(ni, nf, kernel = 3, stride= 1, padding=0):
    return nn.Sequential(
            nn.Conv2d(ni, nf, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU()
    )

class QuoridorNNet(nn.Module):
    """
    Input: 
    Description: 
    Output: None
    """
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = 9, 9
        self.action_size = 64+64+81
        self.args = args

        super(QuoridorNNet, self).__init__()
        self.stem1 = conv_block(1, args.num_channels, padding=1)
        self.stem2 = conv_block(args.num_channels, args.num_channels, padding=1)
        self.conv1 = conv_block(args.num_channels, args.num_channels)
        self.conv2 = conv_block(args.num_channels, args.num_channels)

        self.fc1 = nn.Linear(2+args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, src):
        """
        Arguments:
            src: Tensor, shape [batch_size, 2+81+64]

        Returns:
            output Tensor of shape ````
        """
        sRemWalls = src[:,:2]
        sBoards = src[:,2: 2 + 81]
        s = sBoards.view(-1,self.board_x,self.board_y)               # s: batch_size x 9 x 9
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x 9 x 9
        s = self.stem1(s)                                            # batch_size x num_channels x 9 x 9
        s = self.stem2(s)                                            # batch_size x num_channels x 9 x 9
        s = self.conv1(s)                                            # batch_size x num_channels x 7 x 7
        s = self.conv2(s)                                            # batch_size x num_channels x 5 x 5
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))
        s = torch.cat((sRemWalls, s), dim=1)  
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
