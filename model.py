import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from args import parameter_parser
from layers import *
import torch.nn.init as init
from utils import *

args = parameter_parser()



class GAD(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(GAD, self).__init__()

        self.lin = nn.Linear(feat_size, hidden_size)


    def forward(self, x):
        x_lin = self.lin(x)

        x_lin = torch.tanh(x_lin)
        # x_lin = F.normalize(x_lin, p=2, dim=-1)

        return  x_lin














