from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros,reset
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.utils import spmm


class GATConv(MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        negative_slope = 0.2,
        dropout = 0.0,
        edge_dim = None,
        bias = True,
        residual = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.residual = residual

        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, 1 * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], 1 * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], 1 * out_channels, False,
                                  weight_initializer='glorot')

        self.att_src = Parameter(torch.empty(1, 1, out_channels))
        self.att_dst = Parameter(torch.empty(1, 1, out_channels))

        self.register_parameter('att_edge', None)

        total_out_channels = out_channels * 1

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x,
        edge_index,
        size = None,
        return_attention_weights = None,
    ) :


        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None


        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:
            x_src, x_dst = x
            assert x_src.dim() == 2

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        alpha = self.edge_updater(edge_index, alpha=alpha)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j, alpha_i, index, ptr, dim_size):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class GINConv(MessagePassing):
    def __init__(self,
                 nn,
                 eps = 0.,
                 train_eps = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x,
        edge_index,
        size = None,
    ):

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'