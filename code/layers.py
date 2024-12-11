import torch
from torch_sparse import spmm
from torch import nn  
class SparseNGCNLayer(torch.nn.Module):
   
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(SparseNGCNLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.weight_matrix = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.zeros_(self.bias)

    def define_parameters(self):
      
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)

        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        torch.nn.init.zeros_(self.bias)  

    def forward(self, normalized_adjacency_matrix, features):
      

        if isinstance(normalized_adjacency_matrix, dict):
            indices = normalized_adjacency_matrix['indices'].to(self.device)
            values = normalized_adjacency_matrix['values'].to(self.device)
            size = (features.size(0), features.size(0))  
            normalized_adjacency_matrix = torch.sparse_coo_tensor(indices, values, size).to_dense().to(self.device)

        base_features = torch.mm(features, self.weight_matrix)
        base_features = base_features + self.bias  
        base_features = torch.nn.functional.dropout(base_features, p=self.dropout_rate, training=self.training)
        base_features = torch.nn.functional.relu(base_features)

        for _ in range(self.iterations - 1):
            base_features = torch.mm(normalized_adjacency_matrix, base_features)

        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class DenseNGCNLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):


        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)


    def forward(self, normalized_adjacency_matrix, features):

        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'
