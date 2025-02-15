"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, alpha = None, layerth = None, decoder = False):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.alpha = alpha
        self.layerth = layerth
        self.decoder = decoder

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, v0 = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        src = self.alpha*(v0 - v) if v0 is not None else 0.
        x = (attn @ v + src).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        

        if self.layerth == 0:
            return x, attn, v
        else: return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, alpha = None, layerth = None, decoder = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, alpha= alpha, layerth= layerth, decoder=decoder)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layerth = layerth
        self.decoder = decoder

    def forward(self, x, mask=None, return_attention=False, v0 = None):
        if self.layerth == 0:
            y, attn, v0 = self.attn(self.norm1(x), mask)
        else:
            assert v0 is not None
            y, attn = self.attn(self.norm1(x), mask, v0 = v0)
        if return_attention:
            return attn
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.layerth == 0: 
            assert v0 is not None
            return x, v0
        else: return x