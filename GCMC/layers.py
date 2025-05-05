import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from utils import stack


class RGCLayer(MessagePassing):
    def __init__(self, config, weight_init):
        super(RGCLayer, self).__init__()
        self.in_c = config.num_nodes
        self.out_c = config.hidden_size[0]
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.num_item = config.num_nodes - config.num_users
        self.drop_prob = config.drop_prob
        self.weight_init = weight_init
        self.accum = config.accum
        self.bn = config.rgc_bn
        self.relu = config.rgc_relu
        self.num_features = config.num_features
        
        if self.num_features == 1:
          ord_basis = [nn.Parameter(torch.Tensor(1, self.in_c * self.out_c)) for r in range(self.num_relations)]
        else:
          # ordinal basis matrices 특징벡터 * hidden layer
          ord_basis = [nn.Parameter(torch.Tensor(1, self.num_features * self.out_c)) for r in range(self.num_relations)]
        self.ord_basis = nn.ParameterList(ord_basis)
        self.relu = nn.ReLU()

        if config.accum == 'stack':
            self.bn = nn.BatchNorm1d(self.in_c * config.num_relations)
        else:
            self.bn = nn.BatchNorm1d(self.in_c)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        for basis in self.ord_basis:
            weight_init(basis, self.num_features, self.out_c)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate(self.accum, edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['stack', 'add', 'mean', 'max'] # mean, max 제외
        kwargs['edge_index'] = edge_index 
        size = None
        message_args = []
        for arg in ['x_j', 'edge_type', 'edge_norm', 'edge_index']:
            if arg[-2:] == '_i':
                # tmp is x
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])
                
        update_args = [kwargs[arg] for arg in []]
        
        out = self.message(*message_args)
        if aggr == 'stack':
            out = stack(out, edge_index[0], kwargs['edge_type'], dim_size=size)
        else:
            out = scatter_add(out, edge_index[0], dim=0, dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j, edge_type, edge_norm, edge_index):
        # create weight using ordinal weight sharing
        for relation in range(self.num_relations):
            if relation == 0:
                weight = self.ord_basis[relation]
            else:
                weight = torch.cat((weight, weight[-1] 
                    + self.ord_basis[relation]), 0)

        # weight (R x (in_dim * out_dim)) reshape to (R * in_dim) x out_dim
        # weight has all nodes features
        weight = weight.reshape(-1, self.out_c)

        # feature vector x weight
        if self.num_features != 1:
            splited_weights = torch.split(weight, int(weight.size()[0]/10), dim=0)
                
            for relation in range(self.num_relations):
                if relation == 0:
                    cat_x = x_j.matmul(splited_weights[relation])
                else:
                    cat_x = torch.cat([cat_x, x_j.matmul(splited_weights[relation])],dim=0)
              
        # index has target features index in weitht matrix
            
        idx=edge_index[1]
        index = edge_type * self.in_c + idx

        if self.num_features == 1:
          weight = self.node_dropout(weight)
        else:
          weight = self.node_dropout(cat_x)
        
        out = weight[index]
        # 1/Cj x weight
        return out if edge_norm is None else out * edge_norm.reshape(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        if self.bn:
            aggr_out = self.bn(aggr_out.unsqueeze(0)).squeeze(0)
        if self.relu:
            aggr_out = self.relu(aggr_out)
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        drop_mask = torch.cat([drop_mask for r in range(self.num_relations)], dim=0,).unsqueeze(1)
        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        drop_mask = drop_mask.to(device)
        assert weight.shape == drop_mask.shape
        weight = weight * drop_mask

        return weight


# Second Layer of the Encoder
class DenseLayer(nn.Module):
    def __init__(self, config, weight_init, bias=False):
        super(DenseLayer, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        self.bn = config.dense_bn
        self.relu = config.dense_relu
        self.weight_init = weight_init

        self.dropout = nn.Dropout(config.drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)
        if config.accum == 'stack':
            self.bn_u = nn.BatchNorm1d(config.num_users * config.num_relations)
            self.bn_i = nn.BatchNorm1d((
                config.num_nodes - config.num_users) * config.num_relations
            )
        else:
            self.bn_u = nn.BatchNorm1d(config.num_users)
            self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users)
        self.relu = nn.ReLU()

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(
                    u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(
                    i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)
        return u_features, i_features
