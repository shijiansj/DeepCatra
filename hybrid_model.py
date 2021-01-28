import torch
import torch.nn as nn
from fusion_gpu import LSTM_net, sum_net


class Xi(nn.Module):
    def __init__(self, ln, le, s):
        super(Xi, self).__init__()
        self.ln = ln  # 节点特征向量的维度
        self.le = le  # 边特征向量的维度
        self.s = s  # 状态向量维度

        # 线性网络层
        self.linear = nn.Linear(in_features=2 * ln + le,
                                out_features=s ** 2,
                                bias=True)
        # 激活函数
        self.tanh = nn.Tanh()

    def forward(self, X):
        bs = X.size()[0]
        out = self.linear(X)
        out = self.tanh(out)
        return out.view(bs, self.s, self.s)


# 实现Rou函数
# Input : (N, ln)
# Output : (N, S)
class Rou(nn.Module):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.linear = nn.Linear(in_features=ln,
                                out_features=s,
                                bias=True)
        self.tanh = nn.Tanh()

    def forward(self, X):
        return self.tanh(self.linear(X))


'''
实现Hw函数，即信息生成函数
Initialize :
Input :
    ln : (int)节点特征向量维度
    s : (int)节点状态向量维度
    mu : (int)设定的压缩映射的压缩系数
Forward :
Input :
    X : (Tensor)每一行为一条边的两个节点特征向量以及边的特征向量连接起来得到的向量，shape为(N, 2*ln+le)
    H : (Tensor)与X每行对应的source节点的状态向量
    dg_list : (list or Tensor)与X每行对应的source节点的度向量
Output :
    out : (Tensor)Hw函数的输出
'''


class Hw(nn.Module):
    def __init__(self, ln, le, s, mu=0.9):
        super(Hw, self).__init__()
        self.ln = ln
        self.le = le
        self.s = s
        self.mu = mu

        # 初始化网络层
        self.Xi = Xi(ln, le, s)
        self.Rou = Rou(ln, s)

    def forward(self, X, neis_embeds, H, dg_list):
        if type(dg_list) == list:
            dg_list = torch.Tensor(dg_list)
        elif isinstance(dg_list, torch.Tensor):
            pass
        else:
            raise TypeError("==> dg_list should be list or tensor, not {}".format(type(dg_list)))
        A = (self.Xi(X) * self.mu / self.s) / dg_list.view(-1, 1, 1)  # (N, S, S)
        #b = self.Rou(torch.chunk(X, chunks=2, dim=1)[0])  # (N, S)
        #neis_embeds = torch.float(neis_embeds)
        dest_embeds = neis_embeds.float()
        b = self.Rou(dest_embeds)
        out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)), -1) + b  # (N, s, s) * (N, s) + (N, s)
        return out  # (N, s)


class AggrSum(nn.Module):
    def __init__(self):
        super(AggrSum, self).__init__()


    def forward(self, H, X_neis,V):  #####################x_neis
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_neis] * V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, V - 1).float(), 1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)


# 实现GNN模型
class OriLinearGNN(nn.Module):
    def __init__(self,  feat_dim, stat_dim, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T

        # 点的最终表示经过该层变为一个和图的关联度
        self.linear1 = nn.Linear(in_features= stat_dim,
                                out_features=1,
                                bias=True)
        self.linear2 = nn.Linear(in_features= stat_dim,
                                 out_features=2,
                                 bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        # 实现Fw
        self.Hw = Hw(feat_dim, self.T, stat_dim)

        # 实现H的分组求和
        self.Aggr = AggrSum()

    def forward(self, feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list):
        X_Node = X_Node.long()
        X_Node_decrease = torch.sub(X_Node,1)
        X_Neis = X_Neis.long()
        X_Neis_decrease = torch.sub(X_Neis,1)
        V = feat_Matrix.shape[0]

        node_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Node_decrease)  # (N, ln)
        neis_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Neis_decrease)  # (N, ln)

        edge_type_embeds = torch.zeros((len(edge_type_index), 8))
        for i in range(len(edge_type_index)):
            edge_type_embeds[i][edge_type_index[i] - 1] = 1

        X = torch.cat((node_embeds, neis_embeds, edge_type_embeds), 1)  # (N, 2 * ln+le)

        #H = torch.zeros((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)  # (V, s)
        H = torch.rand((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)  # (V, s),初始状态向量随机初始化
        H = H.to(feat_Matrix.device)
        # 循环T次计算
        for t in range(self.T):
            # (V, s) -> (N, s)
            H = torch.index_select(H, 0, X_Node_decrease)
            #H = torch.index_select(H, 0, X_Node)
            # (N, s) -> (N, s)
            H = self.Hw(X, neis_embeds, H, dg_list)
            # (N, s) -> (V, s)
            H = self.Aggr(H, X_Neis,V)
            # print(H[1])
        # out = torch.cat((feat_Matrix, H), 1)   # (V, ln+s)
        gate_input = torch.cat((H, feat_Matrix),1)
        #print((self.sigmoid(self.linear1(gate_input))).shape)
        soft_attn_weights = self.softmax(self.linear1(H).squeeze())
        node_relation_out = H * soft_attn_weights.unsqueeze(-1)

        graph_out = self.tanh(node_relation_out.sum(dim = 0))
        #print(graph_out)
        #graph_out.view(-1,self.stat_dim)
        #out = self.softmax(self.linear2(graph_out))

        return graph_out  # 最终状态（s,）


class Hybrid_Network(nn.Module):
    def __init__(self, feat_dim, stat_dim, T):
        super(Hybrid_Network, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T



        self.gnn_model = OriLinearGNN(feat_dim, stat_dim, self.T)
        self.lstm = LSTM_net()


        self.linear = nn.Linear(in_features=100,
                                out_features=32,
                                bias=True)
        self.linear = nn.Linear(in_features=64,
                                out_features=2,
                                bias=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list, Lstm_feature):
        gnn_result = self.gnn_model(feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_count = 0
        batch_out = []
        for x_batch in Lstm_feature:
            batch_count += 1
            x = torch.LongTensor(x_batch)
            x = x.to(device)


            out = self.lstm(x)  # torch.Size([64, 128])
            print(out.shape)
            # print(type(out))

            batch_out.append(out)

        net_input = self.tanh(batch_out.sum(dim = 0))
        lstm_out = self.sum(net_input)

        network_out = torch.cat((gnn_result, lstm_out), 0)
        final_out = self.softmax(self.linear(network_out))


        return final_out  # 最终状态（s,）

