__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttentionModel

class MHANetModel(nn.Module):
    def __init__(self, num_channels, steps_ahead, window=24*10, hidRNN=100, hidCNN=100, d_k=64, d_v=64, CNN_kernel=6, highway_window=24, n_head=8, dropout=0.2, rnn_layers=1, output_fun='sigmoid'):
        super(MHANetModel, self).__init__()
        self.window = window
        self.variables = num_channels
        self.hidC = hidCNN
        self.hidR = hidRNN
        self.hw=highway_window

        self.d_v=d_v
        self.d_k=d_k
        self.Ck = CNN_kernel
        self.GRU = nn.GRU(self.variables, self.hidR, num_layers=rnn_layers)
        # self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.variables))

        self.slf_attn = MultiHeadAttentionModel(n_head, self.variables, self.d_k,self.d_v , dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.linear_out=nn.Linear(self.hidR, steps_ahead)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
            self.highway_linear = nn.Linear(num_channels, steps_ahead)
        self.output = None
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (output_fun == 'tanh'):
            self.output = torch.tanh


    def forward(self, x):
        x = x.squeeze(dim=3)
        x = x.permute(0, 2, 1)

        attn_output, slf_attn=self.slf_attn(x,x,x,mask=None)

        r=attn_output.permute(1,0,2).contiguous()
        _,r=self.GRU(r)
        r = self.dropout(torch.squeeze(r[-1:, :, :], 0))
        out = self.linear_out(r)

        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            z = self.highway_linear(z)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out