import torch
import torch.nn as nn

class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln  # 节点特征向量的维度
        self.s = s  # 状态向量维度

        # 线性网络层
        self.linear = nn.Linear(in_features=2 * ln + 10,
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

class Hw(nn.Module):
    def __init__(self, ln, le, s, mu=0.9):
        super(Hw, self).__init__()
        self.ln = ln
        self.le = le
        self.s = s
        self.mu = mu

        # 初始化网络层
        self.Xi = Xi(ln, s)
        self.Rou = Rou(ln, s)

    def forward(self, X, neis_embeds, H, dg_list):
        if type(dg_list) == list:
            dg_list = torch.Tensor(dg_list)
        elif isinstance(dg_list, torch.Tensor):
            pass
        else:
            raise TypeError("==> dg_list should be list or tensor, not {}".format(type(dg_list)))
        A = (self.Xi(X) * self.mu / self.s) / dg_list.view(-1, 1, 1)  # (N, S, S)

        dest_embeds = neis_embeds.float()
        b = self.Rou(dest_embeds)
        out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)), -1) + b  # (N, s, s) * (N, s) + (N, s)
        return out  # (N, s)


class AggrSum(nn.Module):
    def __init__(self):
        super(AggrSum, self).__init__()


    def forward(self, H, X_neis,V):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = torch.stack([X_neis] * V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, V - 1).to(device).float(), 1)
        mask = (mask == 0).float()
        return torch.mm(mask, H)


# 实现GNN模型
class OriLinearGNN(nn.Module):
    def __init__(self,  feat_dim, stat_dim, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T

        self.linear1 = nn.Linear(in_features= stat_dim ,
                                out_features=1,
                                bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        # 实现Fw
        self.Hw = Hw(feat_dim, self.T, stat_dim)

        # 实现H的分组求和
        self.Aggr = AggrSum()
    def forward(self, feat_Matrix, X_Node, X_Neis, edge_type_index, dg_list):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        edge_type_embeds = torch.zeros((edge_type_index.shape[0], 10),dtype=torch.float32)
        for i in range(edge_type_index.shape[0]):
            edge_type_embeds[i][edge_type_index[i] - 1] = 1

        edge_type_embeds=edge_type_embeds.to(device)
        X = torch.cat((node_embeds, neis_embeds, edge_type_embeds), 1)
        X = X.float()
        H = torch.rand((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)
        H = H.to(feat_Matrix.device)
        # 循环T次计算
        for t in range(self.T):
            # (V, s) -> (N, s)
            H = torch.index_select(H, 0, X_Node_decrease)
            # (N, s) -> (N, s)
            H = self.Hw(X, neis_embeds, H, dg_list)
            # (N, s) -> (V, s)
            H = self.Aggr(H, X_Neis,V)
        soft_attn_weights = self.softmax(self.linear1(H).squeeze())
        node_relation_out = H * soft_attn_weights.unsqueeze(-1)

        graph_out = self.tanh(torch.sum(node_relation_out,0))
        return graph_out




