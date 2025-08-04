from typing import Tuple, Union
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, GraphConv
from torch import nn
from dgl.nn.functional import edge_softmax
from functools import partial
import torch.nn.functional as F
import torch
from .layers import RBFExpansion, BesselRBF, GaussianRBF

class MoEHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_scores = self.gate(x)  # [B, E]
        topk_vals, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # [B, k]
    
        topk_weights = F.softmax(topk_vals, dim=-1)  # [B, k]
    
        expert_outputs = []
        for i in range(self.top_k):
            expert_idx = topk_idx[:, i]  # [B]
            expert_weight = topk_weights[:, i].unsqueeze(-1)  # [B, 1]
    
            # 选出当前 expert 对应的结果
            outputs = torch.stack([
                self.experts[expert_idx[j].item()](x[j].unsqueeze(0))
                for j in range(x.size(0))
            ], dim=0).squeeze(1)  # [B, D]
    
            expert_outputs.append(outputs * expert_weight)  # 加权后 shape: [B, D]
    
        output = torch.stack(expert_outputs, dim=0).sum(dim=0)  # [B, D]
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = act_layer()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NodeSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_linear = nn.Linear(dim, dim*3, bias=use_bias)
        self.output = nn.Linear(dim, dim)

    def forward(self, g, h):
        g = g.local_var()
        N = h.shape[0]
        qkv = self.qkv_linear(h).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v

        def edge_attention(edges):
            score = (edges.dst['q'] * edges.src['k']).sum(dim=-1) * self.scale
            return {'score': score}

        g.apply_edges(edge_attention)

        g.edata['attn'] = edge_softmax(g, g.edata['score'])

        def message_func(edges):
            msg = edges.data['attn'].unsqueeze(-1) * edges.src['v']
            return {'m': msg}

        def reduce_func(nodes):
            h_new = nodes.mailbox['m'].sum(dim=1)
            return {'h_new': h_new}

        g.update_all(message_func, reduce_func)

        h_out = g.ndata['h_new'].transpose(1, 2).reshape(N, self.dim)
        return self.output(h_out + h)

class ConditionalAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_linear = nn.Linear(dim, dim*3, bias=use_bias)
        self.c_linear = nn.Linear(dim, dim, bias=use_bias)
        # self.gated = nn.ReLU(inplace=True)
        self.gated = nn.Tanh()

        self.h_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        qkv = self.qkv_linear(h).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        c = self.c_linear(e).reshape(-1, self.num_heads, self.head_dim)
        c = self.gated(c)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v
        g.apply_edges(fn.u_mul_v('k', 'q', 'score'))

        score = g.edata.pop('score') * c
        attn = score.sum(-1, keepdims=True) * self.scale

        g.edata['attn'] = edge_softmax(g, attn)
        g.update_all(fn.u_mul_e('v', 'attn', 'v'), fn.sum('v', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        e_out = self.e_proj(score.reshape(-1, self.dim))

        return h_out, e_out, score.mean(dim=1)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=use_bias)
        self.kv_linear = nn.Linear(dim, dim*2, bias=use_bias)
        self.h_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        q = self.q_linear(h).reshape(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(e).reshape(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        g.ndata['q'] = q
        g.edata['k'] = k
        g.apply_edges(fn.v_dot_e('q', 'k', 'attn'))

        attn = g.edata.pop('attn') * self.scale
        attn = edge_softmax(g, attn)
        g.edata['v'] = attn * v
        g.update_all(fn.copy_e('v', 'm'), fn.sum('m', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        return h_out

class FusiformerLayer(nn.Module):

    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2.0, drop=0.0, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.nodeselfattentiion = NodeSelfAttention(dim, num_heads, use_bias, drop)
        self.norm1 = norm_layer(dim)

        self.crossattention = CrossAttention(dim, num_heads, use_bias, drop)
        self.norm3 = norm_layer(dim)

        self.condattention = ConditionalAttention(dim, num_heads, use_bias, drop)
        self.norm7 = norm_layer(dim)
        self.norm8 = norm_layer(dim)

        self.mlp1 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.mlp2 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.norm9 = norm_layer(dim)
        self.norm10 = norm_layer(dim)

    def forward(self, g, h, e):
        g = g.local_var()

        h = self.norm1(h + self.nodeselfattentiion(g, h))

        h = self.norm3(h + self.crossattention(g, h, e))
        h_, e_, score = self.condattention(g, h, e)

        h = self.norm7(h + h_)
        e = self.norm8(e + e_)

        h = self.norm9(h + self.mlp1(h))
        e = self.norm10(e + self.mlp2(e))

        return (h, e, score)


class FusiformerBlock(nn.Module):

    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2.0, drop=0.0, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dihedral_graph=True):
        super().__init__()

        self.g_update = FusiformerLayer(dim, num_heads, use_bias, mlp_ratio, drop, act_layer, norm_layer)

        self.lg_update = FusiformerLayer(dim, num_heads, use_bias, mlp_ratio, drop, act_layer, norm_layer)

        if dihedral_graph:
            self.dg_update = FusiformerLayer(dim, num_heads, use_bias, mlp_ratio, drop, act_layer, norm_layer)

    def forward(self, g, x, y, lg, z, dg=None, m=None):
        """
        x: node input features
        y: edge input features
        z: edge pair input features
        m: dihedral features
        """
        if dg is not None:
            z, m, score = self.dg_update(dg, z, m)
        y, z, score = self.lg_update(lg, y, z)
        x, y, score = self.g_update(g, x, y)
        return (x, y, z, m)


class Fusiformer(nn.Module):

    def __init__(self, inputs=["graph", "line_graph", "dihedral_graph"], targets=[], depth=4, 
                 edge_input_dim=80, triplet_input_dim=40, embed_dim=128, num_heads=4, 
                 mlp_ratio=2.0, use_bias=True, norm_layer=None, act_layer=None):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.inputs = inputs
        self.targets = targets

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU

        self.atom_embedding = nn.Embedding(95, embed_dim)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_dim),
            nn.Linear(edge_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            nn.Linear(triplet_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        self.dihedral_angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            nn.Linear(triplet_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        self.blocks = nn.ModuleList([
                FusiformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    use_bias=use_bias,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    dihedral_graph='dihedral_graph' in inputs,
                )
                for _ in range(depth)
            ])


        self.readout = AvgPooling()

        self.moe1 = MoEHead(embed_dim, embed_dim, num_experts=4, top_k=2)
        self.moe2 = MoEHead(embed_dim, len(targets), num_experts=4, top_k=2)

    def forward(self, g: Union[(Tuple[(dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph)], Tuple[(dgl.DGLGraph, dgl.DGLGraph)])]):
        """
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata and dg.edata)
        z: angle features (lg.edata)
        m: dihedral_angle features(dg.ndata)
        """

        if isinstance(g, list):
            assert len(g) == 3
            g, lg, dg = g
            g = g.local_var()
            lg = lg.local_var()
            dg = dg.local_var()
            m = self.dihedral_angle_embedding(dg.edata.pop("dihedral_angle"))
        else:
            g, lg = g
            g = g.local_var()
            lg = lg.local_var()
            dg = None
            m = None

        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop("distance"))
        z = self.angle_embedding(lg.edata.pop("angle"))

        for block in self.blocks:
            x, y, z, m = block(g, x, y, lg, z, dg, m)

        h = self.readout(g, x)
        h = self.moe1(h)
        out = self.moe2(h)
        return out

