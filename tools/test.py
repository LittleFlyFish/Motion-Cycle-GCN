import torch
import numpy
from engineer.utils import loss_funcs
from engineer.utils import data_utils as data_utils
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd
import os
from torch.nn.modules.transformer import Transformer
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, frame, D = 10, 16, 66

a = numpy.zeros([13,4,5])
print(a.shape)

# Create random Tensors to hold inputs and outputs
x = torch.randn((N, frame, D), device="cuda:0")
y = torch.randn((N, frame, D), device="cuda:0")

# Use the nn package to define our model and loss function.
model = Transformer(d_model=66, nhead=66, num_encoder_layers=1)
model.to(device="cuda:0")
loss_fn = torch.nn.MSELoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def seg2whole(seg, dct_n):
    # tranasfer element from K windows back to the dct of whole feature
    # seg [seq_len, batch, 66*3]
    b = seg.size(1)
    segs = torch.split(seg, 1, dim=0)
    frame = []
    whole = torch.zeros([b, 20, 66], device="cuda:0")
    for i in range(len(segs)):
        seg_dct = segs[i].view(b, 66, 3)
        seq_i = data_utils.dct2seq(seg_dct, frame_n=5)
        frame.append(seq_i)
        whole[:, i:i+5, :] = whole[:, i:i+5, :] + seq_i

    whole[:, 5:15, :] = whole[:, 5:15, :]/5
    whole[:, 15:20, :] = whole[:, 15:20, :]
    for i in range(5):
        whole[:, i, :] = whole[:, i, :]/(i+1)
    return whole


dct_n = 15
dim_used = range(1,67)
print(len(dim_used))
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x, y)

    # Compute and print loss.
    #loss = loss_fn(y_pred.transpose(0,1), y.transpose(0,1))
    #outputs = torch.randn((16, 66, 15), device="cuda:0")
    all_seq = torch.randn((16, 20, 96), device="cuda:0")
    #outputs = torch.randn((16, 20, 66), device="cuda:0")
    seg = torch.randn((15, 16, 198), device="cuda:0")
    outputs = seg2whole(seg, 15)
    _, loss = loss_funcs.mpjpe_error_p3d_seq2seq(outputs, all_seq, dct_n, dim_used)
    loss = Variable(loss, requires_grad=True)
    print(loss)

    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()