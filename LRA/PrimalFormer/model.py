import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attentions.attention import Attention, EllipticalAttention

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config, elliptical = False, show_M = False, downsample_size = 0, over_layers = False, attenuation = 1e-3):
        super().__init__()

        self.over_layers = over_layers

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config, elliptical = elliptical, show_M = show_M, downsample_size = downsample_size, over_layers = over_layers, attenuation=attenuation)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])
        
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )
        self.attn_type = config["attn_type"]

    def forward(self, X, mask, Q_last = None, V_last = None):
        if self.attn_type.startswith("primal"):
            out, scores, Lambda = self.mha(self.norm1(X), mask)
            X = self.dropout1(out) + X
            X = self.mlpblock(self.norm2(X)) + X
            return X, scores, Lambda
        else:
            if self.over_layers:
                out, Q_last, V_last =  self.mha(self.norm1(X), mask, Q_last, V_last)
            else:
                out = self.mha(self.norm1(X), mask, Q_last, V_last)
            X = self.dropout1(out) + X
            X = self.mlpblock(self.norm2(X)) + X
            if self.over_layers:
                return X, Q_last, V_last
            else:
                return X

class Model(nn.Module):
    def __init__(self, config, M_positions = [], show_M = False, downsample_size = 0, over_layers = False, attenuation = 1e-3):
        super().__init__()

        self.over_layers = over_layers

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)
        self.attn_type = config["attn_type"]

        if self.tied_weights:
            self.transformer = Transformer(config)
        else:
            for idx in range(self.num_layers):
                if idx in M_positions:
                    # set elliptical attention
                    setattr(self, f"transformer_{idx}", Transformer(config, elliptical = True, show_M = show_M, downsample_size = downsample_size, over_layers = over_layers, attenuation=attenuation))
                else:
                    setattr(self, f"transformer_{idx}", Transformer(config, over_layers=over_layers))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask = None, Q_last = None, V_last = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.attn_type.startswith("primal"):
            score_list = []
            Lambda_list = []
            if self.tied_weights:
                for idx in range(self.num_layers):
                    X, scores, Lambda = self.transformer(X, mask)
                    score_list.append(scores)
                    Lambda_list.append(Lambda)
            else:
                for idx in range(self.num_layers):
                    X, scores, Lambda = getattr(self, f"transformer_{idx}")(X, mask)
                    score_list.append(scores)
                    Lambda_list.append(Lambda)

            X = self.norm(X) * mask[:, :, None]
            return X, score_list, Lambda_list

        else:
            if self.tied_weights:
                for idx in range(self.num_layers):
                    if self.over_layers:
                        X, Q_last, V_last = self.transformer(X, mask, Q_last, V_last)
                    else:
                        X = self.transformer(X, mask, Q_last, V_last) 
            else:
                for idx in range(self.num_layers):
                    if self.over_layers:
                        X, Q_last, V_last = getattr(self, f"transformer_{idx}")(X, mask, Q_last, V_last)
                    else:
                        X = getattr(self, f"transformer_{idx}")(X, mask, Q_last, V_last)

            X = self.norm(X) * mask[:, :, None]

            return X