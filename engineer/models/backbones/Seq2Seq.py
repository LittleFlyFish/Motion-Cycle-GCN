#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
from engineer.models.backbones.Motion_GCN import GraphConvolution, GC_Block
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.autograd import Variable
from engineer.models.common.tgcn import ConvTemporalGraphical
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block


class EncoderRNN(nn.Module):
    """
    paras: input, [seq_len, batch, input_size]
    paras: hidden, [n layers * n directions, batch, hidden_size]
    paras: output, [seq_len, batch, hidden_size]
    """
    def __init__(self, input_size, hidden_size, batch=16):
        super(EncoderRNN, self).__init__()
        #input = [seq_len, batch, input_size]
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

        self.batch = batch

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch, self.hidden_size, device="cuda:0")


class AttnDecoderRNN(nn.Module):
    """
    paras: input, [seq_len, batch, input_size]
    paras: hidden, [n layers * n directions, batch, hidden_size]
    paras: output, [seq_len, batch, hidden_size * n directions]
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_p, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input = [seq_len, batch, input_size]
        print(input.shape)

        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        print(output.shape)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device="cuda:0")


@BACKBONES.register_module
class Seq2Seq(nn.Module):
    """
    paras: input_tensor, [seq_len, batch, input_size]
    paras: target_tensor, [seq_len, batch, target_size]
    paras: output, [seq_len, batch, hidden_size * n directions]
    """
    def __init__(self, input_size, hidden_size, output_size, dropout, device, max_length):
        super(Seq2Seq, self).__init__()

        self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size)
        self.decoder = AttnDecoderRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_p=dropout, max_length=max_length)
        self.device = device

        self.max_length = max_length

        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"


    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        batch = target_tensor.size(1)
        outputs = torch.zeros(target_length, batch, self.encoder.hidden_size, device=self.device)

        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        decoder_input = torch.unsqueeze(target_tensor[0, :, :], dim=0)
        decoder_hidden = encoder_hidden

        for t in range(1, target_length):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
            print(decoder_output.shape)
            print(decoder_hidden.shape)
            outputs = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target_tensor[t] if teacher_force else top1
        return outputs