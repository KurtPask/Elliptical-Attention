import torch
import torch.nn as nn
import torch.nn.functional as F

class TropicalLinear(nn.Module):
    """Tropical linear map using max plus multiplication."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim)
        x_exp = x.unsqueeze(-2)
        W_exp = self.W.unsqueeze(0)
        out = x_exp + W_exp
        y, _ = out.max(dim=-1)
        return y

class TropicalMultiHeadAttn(nn.Module):
    """Multi-head attention using TropicalAttention logic."""
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, use_logsumexp=False, use_tropical_metric=True):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.pre_lnorm = pre_lnorm
        self.use_logsumexp = use_logsumexp
        self.use_tropical_metric = use_tropical_metric

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.q_trop = TropicalLinear(d_head, d_head)
        self.k_trop = TropicalLinear(d_head, d_head)
        self.v_trop = TropicalLinear(d_head, d_head)

        self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model))

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def normalize_tropical(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.lambda_param

    def forward(self, h, attn_mask=None, mems=None, **kwargs):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, dim=-1)

        head_q = self.normalize_tropical(torch.log1p(F.relu(head_q)))
        head_k = self.normalize_tropical(torch.log1p(F.relu(head_k)))
        head_v = self.normalize_tropical(torch.log1p(F.relu(head_v)))

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        q = head_q.permute(1, 0, 2, 3)
        k = head_k.permute(1, 0, 2, 3)
        v = head_v.permute(1, 0, 2, 3)

        B, S_q = q.size(0), q.size(1)
        S_k = k.size(1)
        q = q.reshape(B * self.n_head, S_q, self.d_head)
        k = k.reshape(B * self.n_head, S_k, self.d_head)
        v = v.reshape(B * self.n_head, S_k, self.d_head)

        q = self.q_trop(q)
        k = self.k_trop(k)
        v = self.v_trop(v)

        diff = q.unsqueeze(2) - k.unsqueeze(1)
        max_diff = diff.max(dim=-1).values
        min_diff = diff.min(dim=-1).values
        d_trop = max_diff - min_diff
        attn_scores = -d_trop

        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)
        context = sum_sv.max(dim=2).values

        context = context.reshape(B, self.n_head, S_q, self.d_head)
        context = context.permute(2, 0, 1, 3)
        context = torch.expm1(context)
        attn_vec = context.contiguous().view(S_q, B, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)

        return output
