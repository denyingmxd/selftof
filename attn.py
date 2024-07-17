import torch
import torch.nn as nn
import einops
from typing import Optional, Tuple
from einops import rearrange
Size_ = Tuple[int, int]

from torch.nn import Module, Dropout
import torch.nn.functional as F

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        # import ipdb; ipdb.set_trace()
        # QK = torch.einsum("nlhd,nshd->nlsh", Q, K)
        # QK_raw = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # print(f'Q: {Q.shape}, K: {K.shape}')
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        # return queried_values.contiguous(), QK
        return queried_values.contiguous()

class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.ws = ws
        self.encoder_layer = LoFTREncoderLayer(self.dim, num_heads)

    def forward(self, x, size: Size_, y):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.

        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        # print(f'pad {pad_r} {pad_b}')
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = rearrange(x, 'b (sh ws) (sw ws2) c -> (b sh sw) (ws ws2) c', sh=_h, sw=_w)
        # x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3).reshape(B*_h*_w, self.ws*self.ws, C)

        x = self.encoder_layer(x, x)
        if pad_r > 0 or pad_b > 0:
            x = rearrange(x, '(b sh sw) (ws ws2) c -> b (sh ws) (sw ws2) c', sh=_h, sw=_w, ws=self.ws)
            # x = x.view(B, _h, _w, self.ws, self.ws, C).transpose(2, 3).reshape(B, _h*self.ws, _w*self.ws, C)
            x = x[:, :H, :W, :].contiguous()
            x = rearrange(x, 'b h w c -> b (h w) c')
            # x = x.view(B, -1, C)
        else:
            x = rearrange(x, '(b sh sw) (ws ws2) c -> b (sh ws sw ws2) c', sh=_h, sw=_w, ws=self.ws)
        return x


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.encoder_layer = LoFTREncoderLayer(self.dim, num_heads)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape

        query = x.clone()


        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        query = self.encoder_layer(query, x)
        return query

class TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)

    def forward(self, x, size: Size_,y=None):
        y=None
        x = self.lga(x, size,y)
        x = self.gsa(x, size)

        return x

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)

        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        if x_mask is not None:
            tmp_mask = torch.ones_like(x_mask)
        else:
            tmp_mask = None
        message = self.attention(query, key, value, q_mask=tmp_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        if x_mask is not None:
            message.masked_fill_(~x_mask[:,:,None,None], 0)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message + x