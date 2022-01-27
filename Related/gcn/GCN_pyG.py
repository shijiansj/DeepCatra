import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: 增加自连接到邻接矩阵
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: 对节点的特征矩阵进行线性变换
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        #deg = degree(row, size[0], dtype=x_j.dtype)
        deg = torch.ones((size[0], ))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(4, 26)
        self.conv2 = GCNConv(26, 11)
        self.pool = torch.nn.MaxPool1d(3, stride=3,padding = 1)   #0.82
        #self.pool = torch.nn.AvgPool1d(3, stride=3,padding = 1)   #0.78
        self.linear = torch.nn.Linear(4,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        #x = torch.sigmoid(x)
        x= torch.tanh(x)
        #x= torch.relu(x)
        x = self.conv2(x, edge_index)
        #x = torch.sigmoid(x)
        x = torch.tanh(x)
        #x = torch.relu(x)
        x = x.unsqueeze(0)
        x = self.pool(x)
        x =x.squeeze(0)
        Graph_feature = torch.zeros([1,4])
        for single_x in torch.split(x,26,0):
            single_Graph = torch.sum(single_x,dim=0)
            #print("单个图的最终表示",single_Graph)

            single_Graph = torch.unsqueeze(single_Graph,0)
            #########################################################
            #single_Graph = F.tanh(single_Graph)
            Graph_feature = torch.cat((Graph_feature,single_Graph),0)

        Graph_feature = Graph_feature[torch.arange(Graph_feature.size(0)) != 0]
        Graph_out = self.linear(Graph_feature)
        #Graph_out = torch.sigmoid(Graph_out)

        return F.softmax(Graph_out)

