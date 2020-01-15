#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from engineer.models.builder import build_backbone
from engineer.models.registry import BACKBONES
@BACKBONES.register_module
class Cycle_GCN(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, G,G_verse,G_meta,G_verse_meta):
        super(Cycle_GCN, self).__init__()
        self.G = build_backbone(G)
        self.G_verse = build_backbone(G_verse)

        self.G.load_state_dict(torch.load(G_meta)["state_dict"])
        self.G_verse.load_state_dict(torch.load(G_verse_meta)["state_dict"])

    def g(self,x):
        return self.G(x)

    def g_verse(self,x):
        return self.G_verse(x)