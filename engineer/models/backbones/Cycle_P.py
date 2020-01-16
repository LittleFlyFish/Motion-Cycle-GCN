#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from engineer.models.builder import build_backbone
from engineer.models.registry import BACKBONES
@BACKBONES.register_module
class Cycle_P(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, P,P_verse,P_meta,P_verse_meta):
        super(Cycle_P, self).__init__()
        self.P = build_backbone(P)
        self.P_verse = build_backbone(P_verse)

        self.P.load_state_dict(torch.load(P_meta)["state_dict"])
        self.P_verse.load_state_dict(torch.load(P_verse_meta)["state_dict"])

    def p(self,x):
        return self.P(x)

    def p_verse(self,x):
        return self.P_verse(x)