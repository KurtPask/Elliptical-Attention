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
    def __init__(self, dim, heads, dropout, layerth = None, decoder = False):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.layerth = layerth
        self.decoder = decoder

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 2, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = (
            qkv[0],
            qkv[1],
        )

        if self.layerth==0:
            l = torch.zeros((B,self.heads,N,C // self.heads)).to(torch.device("cuda"), non_blocking=True)
            y = torch.zeros((B,self.heads,N,C // self.heads)).to(torch.device("cuda"), non_blocking=True)

            # mu=N*C/8/k.mean(dim=[0,1]).norm(p=1)
            lambd = 4
            mu=N*C/4/k.norm(p=1,dim=[-1,-2],keepdim=True)

            for i in range(0,5):
                s = k-l+y/mu
                s_less = s.le(-lambd*mu).int()
                s_more = s.ge(lambd*mu).int()
                s = (s-lambd*mu)*s_more + (s+lambd*mu)*s_less
                k2 = k-s-y/mu
                l = (k2 @ k2.transpose(-2, -1)) * self.scale
                l = l.softmax(dim=-1)
                l = l @ v
                y = y+mu*(k-l-s)
            
            s = k-l+y/mu
            s_less = s.le(-lambd*mu).int()
            s_more = s.ge(lambd*mu).int()
            s = (s-lambd*mu)*s_more + (s+lambd*mu)*s_less
            k2 = k-s-y/mu
            attn = (k2 @ k2.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            y = y+mu*(k-x-s)

        else:
            attn = (k @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        
            attn = self.attn_drop(attn)

            # @ is a matrix multiplication
            x = (attn @ v)

        # attn = (k @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, layerth = None, decoder = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, layerth= layerth, decoder = decoder)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layerth = layerth
        self.decoder = decoder

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)

  

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x