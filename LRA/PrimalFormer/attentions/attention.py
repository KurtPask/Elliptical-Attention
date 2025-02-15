import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

class SoftmaxAttention(nn.Module):
    def __init__(self, config, over_layers = False):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.over_layers = over_layers


    def forward(self, Q, K, V, mask, Q_last = None, V_last = None):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        if self.over_layers:
            return X, Q, V
        else:
            return X

class EllipticalAttention(nn.Module):
    def __init__(self, config, downsample_size = 0, show_M = False, over_layers = False, attenuation = 1e-3):
        super().__init__()        
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.scale = 1/math.sqrt(self.head_dim)
        self.downsample_size = downsample_size
        self.show_M = show_M
        self.over_layers = over_layers
        self.attenuation = attenuation

    def compute_W(self, Q, K, V, mask, t = None,):
        if t is None:
            t = 0.5 * self.scale
            seqlen = round(Q.size(2) * self.downsample_size) if self.downsample_size != 0 else Q.size(2)
            #t = math.sqrt(1  / math.log(seqlen))
        
        W = []
        for ei in range(self.head_dim):
            H_hat_plustei, H_hat_mintei = self.compute_H_hat(Q, K, V, ei, mask, t, downsample_size=self.downsample_size)
            W_i = (torch.sum(torch.norm(H_hat_plustei - H_hat_mintei, p = 1, dim = -1), dim = -1) / (2*t*seqlen)).unsqueeze(dim = -1)
            W.append(W_i)

        W = torch.cat(W, dim = -1)
        if self.show_M:
            weights = W[0][0] #first item in batch, first head
            W_std = torch.std(weights)
            self.W = (weights, W_std)

            weights_scaled = weights / torch.max(weights)
            scaled_std = torch.std(weights_scaled)
            self.W_scaled = (weights_scaled, scaled_std)

        
        W = W / torch.max(W, dim = -1, keepdim=True)[0]
        #W = W / torch.mean(W, dim = -1, keepdim=True)
        # [bsize x nhead x dhead]
        W = torch.diag_embed(W)
        
        return W

    def compute_H_hat(self, Q, K, V, ei, mask, t, downsample_size = 0):
        with torch.no_grad():
            Q = Q.detach()
            K = K.detach()
            V = V.detach()

            if downsample_size != 0:
                downsample_num = round(Q.shape[2] * downsample_size)
                random_indices = torch.randperm(Q.shape[2])[:downsample_num]
                 # downsample
                Q = Q[:,:,random_indices,:] 
                K = K[:,:,random_indices,:] 
                V = V[:,:,random_indices,:]
                mask = mask[:, random_indices]

            Q_clone = Q.clone()

            Q[...,ei] = Q[..., ei] + t
            Q_clone[...,ei] = Q_clone[..., ei] - t

            Q_concat = torch.concat((Q.unsqueeze(0), Q_clone.unsqueeze(0)), dim = 0) # [2 x bsize x nhead x qlen x dhead]
            K2 = K.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2 x bsize x nhead x klen x dhead]
            # [2 x bsize x nhead x qlen x dhead] x [2 x bsize x nhead x dhead x klen] -> [2 x bsize x nhead x qlen x klen]
            attn_score_par = Q_concat@K2.transpose(-2, -1) * self.scale
            attn_score_par = attn_score_par - 1e-6* (1 - mask[None, :, None, None, :]) # add additional dimension at front due to parallel 2
            attn = nn.functional.softmax(attn_score_par, dim = -1)
            attn = self.drop_attn(attn)

            H_par = attn@V

            H_hat_plus, H_hat_minus = H_par.chunk(chunks = 2, dim = 0)
            H_hat_plus, H_hat_minus = H_hat_plus.squeeze(0), H_hat_minus.squeeze(0)

        return H_hat_plus, H_hat_minus
    
    def W_over_layers(self, Q, Q_last, V, V_last, delta = None, attenuation = 1e-3):
        with torch.no_grad():
            V = V.detach()
            V_last = V_last.detach()
            Q = Q.detach()
            Q_last = Q_last.detach()

            breakpoint()

            seqlen = V.size(2)
            if delta is None:
                deltas = torch.abs(Q - Q_last) + attenuation #include small term for stability and gradient attenuation
                difference_quotients = (V - V_last) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            else:
                #delta = torch.mean(torch.abs(Q - Q_last))
                difference_quotients = (V-V_last) / delta

                # breakpoint()
                # low_threshold_mask = delta*0.8 <= torch.abs(Q-Q_last)
                # high_threshold_mask = delta*1.2 >= torch.abs(Q-Q_last)
                # threshold_mask = low_threshold_mask & high_threshold_mask
                # deltas = torch.where(threshold_mask, Q-Q_last, torch.tensor(float('nan')))
                # difference_quotients = torch.abs(V - V_last) / deltas
                # W = torch.nanmean(difference_quotients, dim = -2)
            
            W = torch.norm(difference_quotients, p = 1, dim = -2) /seqlen #columnwise average l1 norms, [bsize x nhead x dhead]

            if self.show_M:
                weights = W[0][0] #first item in batch, first head
                W_std = torch.std(weights)
                self.W = (weights, W_std)

                weights_scaled = weights / torch.max(weights)
                scaled_std = torch.std(weights_scaled)
                self.W_scaled = (weights_scaled, scaled_std)

                self.deltas =  torch.max(torch.abs(Q-Q_last)), torch.min(torch.abs(Q-Q_last)), torch.mean(torch.abs(Q - Q_last)), torch.std(torch.abs(Q-Q_last))

            
            W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale
            W = torch.diag_embed(W)
            
        return W

    def forward(self, Q, K, V, mask, Q_last = None, V_last = None):
        if self.over_layers:  
            W = self.W_over_layers(Q, Q_last, V, V_last, delta = 1, attenuation=self.attenuation)
        else:
            W = self.compute_W(Q, K, V, mask)
        QW = Q @ W # [bsize x nhead x qlen x dhead]
        dot = QW @ K.transpose(-2,-1) * self.scale # [bsize x nhead x qlen x klen]

        # if self.show_M:
        #     weights = torch.diag(W[0][0]) #first item in batch, first head
        #     W_std = torch.std(weights)
        #     self.W = (weights, W_std)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        if self.over_layers:
            return X, K, V
        else:
            return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class Attention(nn.Module):
    def __init__(self, config, elliptical = False, show_M = False, downsample_size = 0, over_layers = False, attenuation = 1e-3):
        super().__init__()

        self.elliptical = elliptical
        self.show_M = show_M
        self.downsample_size = downsample_size
        self.over_layers = over_layers
        self.attenuation = attenuation

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        if self.attn_type.startswith("primal"):
            self.max_seq_len = config["max_seq_len"]
            self.low_rank = config["low_rank"]
            self.rank_multi = config["rank_multi"]

            self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
            # the weights should be based on batch sample
            self.We = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_head, self.low_rank * self.rank_multi, self.low_rank)))
            self.Wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_head, self.low_rank * self.rank_multi, self.low_rank)))

            if "cos" in self.attn_type:
                from attention_primal_cos import PrimalCosAttention
                self.attn = PrimalCosAttention(config)
        else:
            self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

            if self.attn_type == "softmax":
                if self.elliptical:
                    self.attn = EllipticalAttention(config, show_M = self.show_M, downsample_size=self.downsample_size, over_layers = self.over_layers, attenuation=self.attenuation)
                else:
                    self.attn = SoftmaxAttention(config, over_layers = self.over_layers)
            elif self.attn_type == "cos":
                from attention_cos import CosAttention
                self.attn = CosAttention(config)
            elif self.attn_type == "none":
                self.attn = NoneAttention(config)
            elif self.attn_type.startswith("linformer"):
                from attention_linformer import LinformerAttention
                self.attn = LinformerAttention(config)
            elif self.attn_type.startswith("reformer"):
                from attention_reformer import LSHAttention
                self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
            elif self.attn_type.startswith("nystrom"):
                from attention_nystrom import NystromAttention
                self.attn = NystromAttention(config)
            elif self.attn_type.startswith("performer"):
                from attention_performer import PerformerAttention
                self.attn = PerformerAttention(config)
            elif self.attn_type.startswith("linear"):
                from attention_linear import LinearAttention
                self.attn = LinearAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask, Q_last = None, V_last = None):
        if self.attn_type.startswith("primal"):
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            # evenly sample
            indices = torch.linspace(0, X.shape[1]-1, self.low_rank * self.rank_multi, dtype=int)
            X = X.transpose(-2,-1).reshape(X.size(0), self.num_head, self.head_dim, X.size(1))
            X = X[:, :, :, indices].transpose(1, 2)

            if "cos" in self.attn_type:
                we = torch.einsum('bahd,hde->bahe', X, self.We.type_as(X)).transpose(1,2)
                wr = torch.einsum('bahd,hde->bahe', X, self.Wr.type_as(X)).transpose(1,2)

            with torch.cuda.amp.autocast(enabled = False):
                attn_out, scores, Lambda = self.attn(Q.float(), K.float(), [we.float(), self.We], [wr.float(), self.Wr], mask.float())
            attn_out = self.combine_heads(attn_out)
            out = self.ff(attn_out)

            return out, scores, Lambda

        else:
            if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
                with torch.cuda.amp.autocast(enabled = False):
                    attn_out = self.attn(X.float(), mask.float())
            else:
                Q = self.split_heads(self.W_q(X))
                K = self.split_heads(self.W_k(X))
                V = self.split_heads(self.W_v(X))
                if Q_last is not None and V_last is not None:
                    Q_last = Q_last.float()
                    V_last = V_last.float()
                    
                with torch.cuda.amp.autocast(enabled = False):
                    if self.grad_checkpointing:
                        attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float(), Q_last, V_last)
                    else:
                        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float(), Q_last.float(), V_last.float())
                if self.over_layers:
                    attn_out, Q_last, V_last = attn_out    
                attn_out = self.combine_heads(attn_out)
            
            out = self.ff(attn_out)

            if self.over_layers:
                return out, Q_last, V_last
            else:
                return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
