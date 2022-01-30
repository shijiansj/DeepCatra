import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score

def encoding():
    opcode_dict={}
    with open("opcodes_encoding.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip("\n")
            if line!="":
                opcode=line[:line.find(":")]
                num=int(line[line.find(":")+1:])
                opcode_dict[opcode]=num
    return opcode_dict
opcode_dict = encoding()

def split_opcode_seq(opcode_seq,split_length):
    opcode_seq_list=[]
    num=int(len(opcode_seq)/split_length)
    for i in range(0,num):
        opcode_seq_list.append(np.asarray(opcode_seq[len(opcode_seq)-(split_length*(i+1)):len(opcode_seq)-(split_length*i)]))
    opcode_seq_list=reversed(opcode_seq_list)
    return opcode_seq_list

def preprocess(graph_vertix,graph_edge):
    for i in range(len(graph_edge)):
        graph_edge[i] = np.unique(graph_edge[i], axis=0)

    Edge_list = []

    for i in range(len(graph_edge)):
        edge_list = defaultdict(list)
        for j in range(graph_edge[i].shape[0]):
            # 反向边
            if (graph_edge[i][j][1] in edge_list):
                edge_list[graph_edge[i][j][1]].append((graph_edge[i][j][0] + 5, graph_edge[i][j][2]))
            else:
                edge_list[graph_edge[i][j][1]] = [(graph_edge[i][j][0] + 5, graph_edge[i][j][2])]
            # 前向边
            if(graph_edge[i][j][2] in edge_list):
                edge_list[graph_edge[i][j][2]].append((graph_edge[i][j][0], graph_edge[i][j][1]))
            else:
                #degree_list.append(n2)
                edge_list[graph_edge[i][j][2]] = [(graph_edge[i][j][0], graph_edge[i][j][1])]

        Edge_list.append(edge_list)


    node_source_list = []
    node_dest_list = []
    edge_type_index_list = []
    dg_list = []

    for x in range(len(graph_edge)):
        node_source = []
        node_dest = []
        edge_type_index = []
        for i in list(sorted(Edge_list[x].keys())):
            for j in list(Edge_list[x][i]):
                node_source.append(j[1])
                edge_type_index.append(j[0])
                node_dest.append(i)


        node_source_list.append(np.int16(node_source))
        node_dest_list.append(np.int16(node_dest))
        edge_type_index_list.append(np.int8(edge_type_index))
    # 生成度向量
    for i in range(len(graph_edge)):
        _,x_unique = np.unique(node_dest_list[i], return_counts=True)

        node_dest_decrease = np.array([x - 1 for x in node_dest_list[i]])
        dg_list.append(np.array(x_unique[node_dest_decrease]))

    return graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list
def load_my_data_split(deal_folder,split_length):

    opcode_dict=encoding()
    feature_data=[]
    with open(deal_folder, "r", encoding="utf-8") as file:

        opcode_seq = []
        for line in file.readlines():
            line = line.strip("\n")
            if line == "" and len(opcode_seq) != 0:
                feature_data.extend(split_opcode_seq(opcode_seq,split_length))
                opcode_seq = []
            elif line.find(":") == -1 and line != "":
                opcode_seq.append(np.int32(opcode_dict[line]))

        if len(opcode_seq) != 0:
            if len(opcode_seq) >= split_length:
                feature_data.extend(split_opcode_seq(opcode_seq,split_length))
    return feature_data

def get_data(path, ln,split_length):

    path_benign = os.path.join(path, 'benign')
    path_malicious = os.path.join(path, 'malware')
    hash_list = os.listdir(path_malicious) + os.listdir(path_benign)
    dir = os.listdir(path)

    graph_vertix = []
    graph_edge = []
    lstm_feature = []
    labels = []

    i = -1
    for files in dir:  # 遍历文件夹
        sub_path = os.path.join(path, files)
        file = os.listdir(sub_path)
        i = i + 1
        print(files)
        for apk in file:
            sub_sub_path = os.path.join(sub_path, apk)
            edge_path = os.path.join(sub_sub_path, 'edge.txt')
            vertix_path = os.path.join(sub_sub_path, 'vertix.txt')
            opcode = 'sensitive_opcode_seq.txt'
            opcode_path = os.path.join( sub_sub_path, opcode)
            vandeando = os.listdir(sub_sub_path)
            if files =='malware':
                labels.append(1)
            else:
                labels.append(0)
            for vande in vandeando:
                if (vande == 'edge.txt'):
                    edge_info = open(edge_path)
                    lines = edge_info.readlines()
                    edge = np.zeros((len(lines), 3), dtype=int)
                    j = 0
                    for line in lines:
                        curline = line.strip('\n')
                        curline = curline.split()
                        curline = [int(i) for i in curline]
                        curline = np.array(curline)
                        edge[j] = curline
                        j += 1
                    graph_edge.append(np.array(edge))

                if (vande == 'vertix.txt'):
                    vertix_info = open(vertix_path)
                    i = 0
                    lines = vertix_info.readlines()
                    vertix = np.zeros((len(lines), ln), dtype=float)

                    for line in lines:
                        curline = line.strip('\n')
                        curline = curline.split()
                        curline = [int(i) for i in curline]

                        if (len(curline) < ln):
                            curline = list(curline + [0] * (ln - len(curline)))
                        if (len(curline) > ln):
                            curline = curline[:ln]
                        curline = np.array(curline)
                        curline = curline.astype(float)

                        # 归一化
                        curline = curline / 232
                        vertix[i] = curline
                        i += 1
                    graph_vertix.append(vertix)

                if(vande == opcode):

                    single_apk_data = load_my_data_split(opcode_path,split_length)
                    single_apk_data = np.array(single_apk_data)
                    lstm_feature.append(np.array(single_apk_data))
    num1=0
    num0=0
    print(len(graph_edge))
    print(len(graph_vertix))
    print(len(labels))
    print(len(lstm_feature))
    for x in labels:
        if x==1:
            num1+=1
        if x==0:
            num0+=1
    print(num1)
    print(num0)
    return labels, graph_vertix, graph_edge, lstm_feature

class LSTM_net(nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        self.embedding = nn.Embedding(len(opcode_dict),128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(512, 64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                 nn.Tanh())
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        x = self.tanh(torch.mean(x,0))
        return x

class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln  # 节点特征向量的维度
        # self.le = le  # 边特征向量的维度
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

    def forward(self, H, X_neis, V):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = torch.stack([X_neis] * V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, V - 1).to(device).float(), 1)
        mask = (mask == 0).float()
        return torch.mm(mask, H)

class OriLinearGNN(nn.Module):
    def __init__(self, feat_dim, stat_dim, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T

        self.linear1 = nn.Linear(in_features=stat_dim,
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
        X_Node_decrease = torch.sub(X_Node, 1)
        X_Neis = X_Neis.long()
        X_Neis_decrease = torch.sub(X_Neis, 1)
        V = feat_Matrix.shape[0]

        node_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Node_decrease)  # (N, ln)
        neis_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Neis_decrease)  # (N, ln)

        edge_type_embeds = torch.zeros((edge_type_index.shape[0], 10), dtype=torch.float32)
        for i in range(edge_type_index.shape[0]):
            edge_type_embeds[i][edge_type_index[i] - 1] = 1

        edge_type_embeds = edge_type_embeds.to(device)
        X = torch.cat((node_embeds, neis_embeds, edge_type_embeds), 1)
        X = X.float()
        H = torch.rand((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)
        H = H.to(feat_Matrix.device)
        for t in range(self.T):
            H = torch.index_select(H, 0, X_Node_decrease)
            H = self.Hw(X, neis_embeds, H, dg_list)
            H = self.Aggr(H, X_Neis, V)
        soft_attn_weights = self.softmax(self.linear1(H).squeeze())
        node_relation_out = H * soft_attn_weights.unsqueeze(-1)

        graph_out = self.tanh(torch.sum(node_relation_out, 0))

        return graph_out

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

def get_split_dataset(path,ln,split_length):

    labels, graph_vertix, graph_edge, lstm_feature = get_data(path, ln,split_length)
    graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list\
        = preprocess(graph_vertix, graph_edge)

    np.random.seed(0)
    indices = np.random.permutation(len(graph_vertix))

    graph_vertix = np.array(graph_vertix, dtype=object)[indices]
    node_source_list = np.array(node_source_list, dtype=object)[indices]
    node_dest_list = np.array(node_dest_list, dtype=object)[indices]
    edge_type_index_list = np.array(edge_type_index_list, dtype=object)[indices]
    dg_list = np.array(dg_list, dtype=object)[indices]
    lstm_feature = np.array(lstm_feature, dtype=object)[indices]

    labels = np.array(labels)[indices]
    dataset = [graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature,labels]
    return dataset

def test(test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 10
    model = Hybrid_Network(13, 32, T)

    model.load_state_dict(torch.load('model_best_params'))
    model.to(device)
    model.eval()
    test_pred = []
    prob_labels = []
    Graph_vertix = test[0]

    Node_source_list = test[1]
    Node_dest_list = test[2]
    Edge_type_index_list = test[3]
    Dg_list = test[4]
    Lstm_feature = test[5]
    labels = test[6]
    with torch.no_grad():
        for i in range(len(Graph_vertix)):
            lstm_feature = Lstm_feature[i].astype(int)
            graph_vertix = Graph_vertix[i].astype(float)
            node_source_list = Node_source_list[i].astype(int)
            node_dest_list = Node_dest_list[i].astype(int)
            edge_type_index_list = Edge_type_index_list[i].astype(int)
            dg_list = Dg_list[i].astype(int)

            lstm_feature = torch.LongTensor(lstm_feature)
            graph_vertix = torch.FloatTensor(graph_vertix)
            node_source_list = torch.LongTensor(node_source_list)
            node_dest_list = torch.LongTensor(node_dest_list)
            edge_type_index_list = torch.LongTensor(edge_type_index_list)
            dg_list = torch.LongTensor(dg_list)

            lstm_feature = lstm_feature.to(device)
            graph_vertix = graph_vertix.to(device)
            node_source_list = node_source_list.to(device)
            node_dest_list = node_dest_list.to(device)
            edge_type_index_list = edge_type_index_list.to(device)
            dg_list = dg_list.to(device)


            out = model(graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature)
            pred = torch.max(out, 1)[1].cpu().numpy()
            prob_label = out.cpu().numpy()
            prob_labels.append(prob_label[0][0])
            test_pred.append(pred[0])


        test_pred = np.array(test_pred)
        accuracy = accuracy_score(test[6], test_pred)
        precision = precision_score(test[6], test_pred, average='binary')  # 输出精度
        recall = recall_score(test[6], test_pred, average='binary')  # 输出召回率
        f1 = f1_score(test[6], test_pred, average='binary')
        auc = roc_auc_score(test[6],prob_labels)
        print('accuracy: ', accuracy)
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1-score: ', f1)
        print('auc: ', auc)


def main():
    test_dataset_path = sys.argv[1]
    testdataset = get_split_dataset(test_dataset_path,13,100)
    test(testdataset)

if __name__ == "__main__":
    main()

