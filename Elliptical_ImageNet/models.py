import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Attention, Block
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

#import fourier_layer_cuda

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
    'deit_fourier_tiny_patch16_224'
]
    
class FourierAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.paramR = nn.Parameter(1.0 * torch.ones(1), requires_grad= True) 

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        q = q.permute(2, 0, 1, 3)
        k = k.permute(2, 0, 1, 3)
        attn = FOURIERFunction.apply( q.float().contiguous(), k.float().contiguous(), self.paramR)
        attn = attn**4
        attn = attn / ((torch.sum(attn, dim=1))[:, None, :, :] + 1e-6)
        attn = attn.permute(2, 3, 0, 1)
        attn = self.attn_drop(attn)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MahalaAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0., M = True, t = None, downsample_size = 0, show_M = False,
                 simultaneous_comp = False, over_layers = False, attenuation = 1e-3, ablation = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.q_net = nn.Linear(dim, dim, bias = False)
        #self.kv_net = nn.Linear(dim, 2*dim, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.M = M
        self.t = t
        self.downsample_size = downsample_size
        self.W = (torch.zeros(head_dim), 0.0)
        self.W_scaled = (torch.zeros(head_dim), 0.0)
        self.deltas = (0.0, 0.0, 0.0, 0.0)
        self.diff_quotients = (0.0, 0.0, 0.0, 0.0)
        self.show_M = show_M
        self.simultaneous_comp = simultaneous_comp
        self.over_layers = over_layers
        self.attenuation = attenuation
        self.ablation = ablation

        #self.paramR = nn.Parameter(1.0 * torch.ones(1), requires_grad= True)

    def compute_W(self, head_q, head_k, head_v, t = None):
        if t is None:
            t = 0.5 * (self.scale)
        W = []
        for ei in range(self.head_dim):
            H_hat_plustei, H_hat_mintei = self.compute_H_hat(head_q, head_k, head_v, ei, t=t, downsample_size = self.downsample_size)
            #H_hat_plustei = self.compute_H_hat(head_q, head_k, head_v, ei, t = t, plus_minus= "plus", downsample_size= self.downsample_size)
            #H_hat_mintei = self.compute_H_hat(head_q, head_k, head_v, ei, t = t, plus_minus= "minus", downsample_size= self.downsample_size)
            # [bsz x nhead x 1]
            # (H_hat_plustei - H_hat_minustei): [bsize, nhead, qlen, dhead]
            W_i = (torch.sum(torch.norm(H_hat_plustei - H_hat_mintei, p = 1, dim = -1), dim = -1) / (2*t*head_q.size(2))).unsqueeze(dim = -1)
            W.append(W_i)
        
        # ### experimental purposes only, ensure not executing during actual training ###
        # if self.compare_downsample_grads:
        #     W_full = []
        #     for ei in range(self.d_head):
        #         H_hat_plustei = self.compute_H_hat(head_q, head_k, head_v, ei, t = t, plus_minus= "plus", downsample_size= 0)
        #         H_hat_mintei = self.compute_H_hat(head_q, head_k, head_v, ei, t = t, plus_minus= "minus", downsample_size= 0)
        #         # [bsz x nhead x 1]
        #         W_i = (torch.sum(torch.norm(H_hat_plustei - H_hat_mintei, p = 1, dim = -1), dim = 0) / (2*t*head_q.size(0))).unsqueeze(dim = -1)
        #         W_full.append(W_i)
        #     # [bsz x nhead x dhead]
        #     W_full = torch.cat(W_full, dim = -1)
        #     W_full = torch.diag_embed(W_full)
        #     self.W_full = torch.diag(W_full[0][0])
        # ### experimental purposes only, ensure not executing during actual training ###
        # [bsz x nhead x dhead]
        W = torch.cat(W, dim = -1)
        # [bsz x nhead x dhead x dhead]
        W = torch.diag_embed(W)

        return W
    
    def compute_H_hat(self, head_q, head_k, head_v, ei, t, attn_mask=None, plus_minus = "plus", downsample_size = 0):
        with torch.no_grad():
            head_q = head_q.detach() # [bsize x nhead x qlen x dhead]
            head_k = head_k.detach() # [bsize x nhead x klen x dhead]
            head_v = head_v.detach() # [bsize x nhead x vlen x dhead]

            # clones for parallel comp
            head_q_clone = head_q.clone()
            # head dim: [qlen x bsize x nhead x dhead]
            # downsampling to be deprecated, consider removing ###
            # if downsample_size != 0: # 0 for no downsampling
            #     random_indices = torch.randperm(head_q.shape[0])[:downsample_size]
            #     # downsample
            #     head_q = head_q[random_indices] 
            #     head_k = head_k[random_indices] 
            #     head_v = head_v[random_indices]

            head_q[..., ei] = head_q[..., ei] + t # positive peturb
            head_q_clone[..., ei] = head_q_clone[..., ei] - t # negative peturb

            head_q_concat = torch.concat((head_q.unsqueeze(0), head_q_clone.unsqueeze(0)), dim = 0) # [2 x bsize x nhead x qlen x dhead]
            head_k2 = head_k.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2 x bsize x nhead x klen x dhead]
            #head_v2 = head_v.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2 x bsize x nhead x vlen x dhead]  #don't need as torch will broadcast it

            #attn_score_par = torch.einsum('cqbnd, ckbnd -> cqkbn', (head_q_concat, head_k2)) # [2 x qlen x klen x bsize x nhead]
            # [2 x bsize x nhead x qlen x dhead] x [2 x bsize x nhead x dhead x klen] -> [2 x bsize x nhead x qlen x klen]
            attn_score_par = head_q_concat@head_k2.transpose(-2, -1) * self.scale
            #attn_score_par.mul_(self.scale)
            #attn_score_par = attn_score_par * self.scale
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_score_par.masked_fill_(
                        #attn_mask[None,:,:,None], -float('inf'))
                        attn_mask[:, None,:,:,None], -float('inf')) # skipping 0th dimension to avoid the 2 from parallel computation
                elif attn_mask.dim() == 3:
                    #attn_score_cat.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
                    attn_score_par.masked_fill_(attn_mask[:,:,:,:,None], -float('inf')) # skipping 0th dimension again

            
            attn_prob_par = F.softmax(attn_score_par, dim = -1)
            attn_prob_par = self.attn_drop(attn_prob_par)
            # [qlen x bsize x n_head x 2d_head]
            #H_par = torch.einsum('cqkbn,ckbnd->cqbnd', (attn_prob_par, head_v2))
            
            # [2 x bsize x nhead x qlen x klen] x [2 x bsize x nhead x vlen x dhead] -> [2 x bsize x nhead x qlen x dhead]
            H_par = attn_prob_par@head_v
            #H_par = attn_prob_par@head_v2
        
            H_hat_plus, H_hat_minus = H_par.chunk(chunks = 2, dim = 0)
            H_hat_plus, H_hat_minus = H_hat_plus.squeeze(0), H_hat_minus.squeeze(0)
    
        return H_hat_plus, H_hat_minus

    
    def simultaneous_grads(self, head_q, head_k, head_v, t, attn_mask = None):
        with torch.no_grad():
            head_q = head_q.detach() # [bsize x nhead x qlen x dhead]
            head_k = head_k.detach() # [bsize x nhead x klen x dhead]
            head_v = head_v.detach() # [bsize x nhead x vlen x dhead]

            # head_q[..., ei] = head_q[..., ei] + t # positive peturb
            # head_q_clone[..., ei] = head_q_clone[..., ei] - t # negative peturb

            # simultaneously perturb all dimensions
            SP = torch.ones(0, 2, (head_q.size(2), head_q.size(3))) * t
            head_q_plus = head_q + SP
            head_q_minus = head_q - SP

            # attention forward
            head_q_concat = torch.concat((head_q_plus.unsqueeze(0), head_q_minus.unsqueeze(0)), dim = 0) # [2 x bsize x nhead x qlen x dhead]
            head_k2 = head_k.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2 x bsize x nhead x klen x dhead]

            #attn_score_par = torch.einsum('cqbnd, ckbnd -> cqkbn', (head_q_concat, head_k2)) # [2 x qlen x klen x bsize x nhead]
            # [2 x bsize x nhead x qlen x dhead] x [2 x bsize x nhead x dhead x klen] -> [2 x bsize x nhead x qlen x klen]
            attn_score_par = head_q_concat@head_k2.transpose(-2, -1) * self.scale
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_score_par.masked_fill_(
                        #attn_mask[None,:,:,None], -float('inf'))
                        attn_mask[:, None,:,:,None], -float('inf')) # skipping 0th dimension to avoid the 2 from parallel computation
                elif attn_mask.dim() == 3:
                    #attn_score_cat.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
                    attn_score_par.masked_fill_(attn_mask[:,:,:,:,None], -float('inf')) # skipping 0th dimension again

            
            attn_prob_par = F.softmax(attn_score_par, dim = -1)
            attn_prob_par = self.attn_drop(attn_prob_par)
            # [qlen x bsize x n_head x 2d_head]
            
            # [2 x bsize x nhead x qlen x klen] x [2 x bsize x nhead x vlen x dhead] -> [2 x bsize x nhead x qlen x dhead]
            H_plus, H_minus = torch.chunk(attn_prob_par@head_v, chunks =2, dim = 0)
            W = torch.norm((H_plus.squeeze(0) - H_minus.squeeze(0)), p = 1, dim = -2) # [bsize x nhead x dhead]
            W = torch.diag_embed(W) # [bsize x nhead x dhead x dhead]

            return W
    
    def W_over_layers(self, head_q, head_q_last, head_v, head_v_last, delta = None, attenuation = 1e-3):
        with torch.no_grad():
            head_v = head_v.detach()
            head_v_last = head_v_last.detach()
            head_q = head_q.detach()
            head_q_last = head_q_last.detach()


            seqlen = head_v.size(2)
            if delta is None:
                deltas = torch.abs(head_q - head_q_last) + attenuation #include small term for stability and gradient attenuation
                difference_quotients = (head_v - head_v_last) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            else:
                #delta = torch.mean(torch.abs(head_q - head_q_last))
                difference_quotients = (head_v-head_v_last) / delta

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

                self.deltas =  torch.max(torch.abs(head_q-head_q_last)), torch.min(torch.abs(head_q-head_q_last)), torch.mean(torch.abs(head_q - head_q_last)), torch.std(torch.abs(head_q-head_q_last))
                self.diff_quotients =  torch.max(torch.abs(head_v-head_v_last)), torch.min(torch.abs(head_v-head_v_last)), torch.mean(torch.abs(head_v - head_v_last)), torch.std(torch.abs(head_v-head_v_last))
            
            W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale

            W = torch.diag_embed(W)
            
        return W

    def W_ablation(self, b_size, bottom = 0.):
        with torch.no_grad():
            # bottom controls the max stretch of any one dimension
            W = (1-bottom)*torch.rand(b_size, self.num_heads, self.head_dim) + bottom
            W = W / torch.max(W, dim = -1, keepdim=True)[0]
            W = torch.diag_embed(W)

        return W
            
    def rep_collapse(self, x):
        #breakpoint()
        n = x.shape[1]
        x_norm = torch.norm(x, 2, dim = -1, keepdim = True)
        x_ = x / x_norm
        x_cossim = torch.tril((x_ @ x_.transpose(-2,-1 )), diagonal = -1).sum(dim = (-1,-2)) / (n*(n-1)/2)
        x_cossim = x_cossim.mean()
        self.cossim = x_cossim

        return
       
    def forward(self, x, head_q_last = None, head_v_last = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        head_q, head_k, head_v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        #breakpoint()
        if self.M: # mahlattention computation

            if self.ablation:
                W = self.W_ablation(B)
                W = W.to(head_q.device)
            elif self.simultaneous_comp:
                W = self.simultaneous_grads(head_q, head_k, head_v, t = self.t)
            elif self.over_layers:
                W = self.W_over_layers(head_k, head_q_last, head_v, head_v_last, delta = 5, attenuation=self.attenuation) # in this alternate version, head_q_last coming in as actually head_k_last, just bad naming
            else:
                W = self.compute_W(head_q, head_k, head_v, t = self.t) #[bsize x nhead x dhead x dhead]
            
            QW = head_q @ W # [bsize x nhead x qlen x dhead]
            #attn_score = torch.einsum('ibnd,jbnd->ijbn', (QW, head_k))
            attn_score = QW @ head_k.transpose(-2,-1) * self.scale # [bsize x nhead x qlen x klen]

        else: # normal attention computation
            attn_score = head_q @ head_k.transpose(-2,-1) * self.scale
        
        attn_prob = F.softmax(attn_score, dim = -1) 
        attn_prob = self.attn_drop(attn_prob)
        
        x = (attn_prob @ head_v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #self.rep_collapse(x)
        
        if self.over_layers:
            #return x, head_q, head_v
            return x, head_k, head_v
        else:
            return x
    
class FOURIERFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_q, head_k, paramR):
        #print('type(head_q): ', head_q.type());
        #print('type(head_k): ', head_k.type());
        #print('type(paramR): ', paramR.type());
        output = fourier_layer_cuda.forward(head_q, head_k, paramR)
        variables = head_q, head_k, paramR, output
        #print('size(head_q) = ',head_q.size());
        #print('size(head_k) = ',head_k.size());
        #print('size(paramR) = ',paramR.size());
        #print('size(output) = ',output.size());
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad):
        head_q, head_k, paramR, output = ctx.saved_tensors
        grads = fourier_layer_cuda.backward(
            grad.contiguous(), head_q, head_k, paramR, output)
        grad_head_q, grad_head_k, grad_p = grads
        return grad_head_q, grad_head_k, grad_p
    
    
class FourierBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MahalaBlock(nn.Module):

    def __init__(self, dim, num_heads, M, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, show_M = False, over_layers = False, attenuation = 1e-3, ablation = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MahalaAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
                                    M = M, show_M = show_M, over_layers = over_layers, attenuation = attenuation, ablation = ablation)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.over_layers = over_layers

    def forward(self, x, head_q_last = None, head_v_last = None):
        if self.over_layers:
            if isinstance(x, tuple):
                x, head_q_last, head_v_last = x # unpack tuple
            attn_out, head_q_last, head_v_last = self.attn(self.norm1(x), head_q_last, head_v_last)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.over_layers:
            return x, head_q_last, head_v_last
        else:
            return x

class FourierVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            FourierBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

class MahalaVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', M_positions = [], show_M = False, over_layers = False, attenuation = 1e-3, ablation = False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.M_positions = M_positions
        self.over_layers = over_layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MahalaBlock( # append a mahalattention block
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, M = True, show_M = show_M, over_layers=over_layers, attenuation=attenuation,
                ablation = ablation) if i in M_positions else
            MahalaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, M = False, over_layers=over_layers) # append regular attention block   
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        if self.over_layers:
            x, head_q_last, head_v_last = x
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        
        
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_mahala_tiny_patch16_224(pretrained=False, **kwargs):
    model = MahalaVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_mahala_overlayers_tiny_patch16_224(pretrained=False, **kwargs):
    model = MahalaVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, over_layers= True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_mahala_overlayers_tiny2_patch16_224(pretrained=False, **kwargs):
    model = MahalaVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, over_layers= True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_fourier_tiny_patch16_224(pretrained=False, **kwargs):
    model = FourierVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
