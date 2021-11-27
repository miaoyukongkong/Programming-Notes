"""通过nn.Sequential()快速搭建三层感知机

File name: mlp.py
Author: @dongdonghy
Date: 2021-11-28

"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        super(MLP, self).__init__()
        #通过nn.Sequential()快速搭建三层感知机
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim1),
          nn.ReLU(),
          nn.Linear(hid_dim1, hid_dim2),
          nn.ReLU(),
          nn.Linear(hid_dim2, out_dim),
          nn.ReLU()
       )
    def forward(self, x):
        x = self.layer(x)
        return x
