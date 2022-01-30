import torch
import torch.nn as nn
from lstm_model import LSTM_net
from gnn_model import OriLinearGNN

class Hybrid_Network(nn.Module):
    def __init__(self, feat_dim, stat_dim,T):
        super(Hybrid_Network, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T

        self.gnn_model = OriLinearGNN(feat_dim, stat_dim, self.T)
        self.lstm = LSTM_net()


        self.linear1 = nn.Linear(in_features=64,
                                out_features=32,
                                bias=True)
        self.linear = nn.Linear(in_features=64,
                                out_features=2,
                                bias=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list, Lstm_feature):
        gnn_result = self.gnn_model(feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list)
        lstm_out = self.lstm(Lstm_feature)
        network_out = torch.cat([gnn_result.view(-1,32), lstm_out.view(-1,32)], 1)
        final_out = self.softmax(self.linear(network_out))
        return final_out




