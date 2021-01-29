import os
import scipy
import numpy as np
import networkx as nx
from collections import defaultdict
import torch.nn.functional  as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from new_gnn_model import OriLinearGNN
from data_reader import load_data
from dataloader import GraphDataset
from sklearn.model_selection import ShuffleSplit

def __eq__(x, y):
    return x.__dict__ == y.__dict__



def preprocess(graph_vertix,graph_edge,labels):

    #nodes_num_list = v_matrix.shape[0]
    #去重
    for i in range(len(graph_edge)):
        graph_edge[i] = np.unique(graph_edge[i],axis = 0)


    # 首先统计得到节点的度
    Degree_list = []  # type: List[List[Tuple[int, int]]]

    for i in range(len(graph_edge)):
        #degree_list = dict()
        degree_list = defaultdict(list)
        for j in range(graph_edge[i].shape[0]):
            # 反向边
            if (graph_edge[i][j][1] in degree_list):
                degree_list[graph_edge[i][j][1]].append((graph_edge[i][j][0] + 4, graph_edge[i][j][2]))
            else:
                # degree_list.append(n1)
                degree_list[graph_edge[i][j][1]] = [(graph_edge[i][j][0] + 4, graph_edge[i][j][2])]
            # 前向边
            if(graph_edge[i][j][2] in degree_list):
                degree_list[graph_edge[i][j][2]].append((graph_edge[i][j][0], graph_edge[i][j][1]))
            else:
                #degree_list.append(n2)
                degree_list[graph_edge[i][j][2]] = [(graph_edge[i][j][0], graph_edge[i][j][1])]

        Degree_list.append(degree_list)
    # 然后生成两个向量



    node_source_list = []
    node_dest_list = []
    edge_type_index_list = []
    dg_list = []

    for x in range(len(graph_edge)):
        node_source = []
        node_dest = []
        edge_type_index = []
        for i in list(Degree_list[x].keys()):
            for j in list(Degree_list[x][i]):
                node_source.append(j[1])
                edge_type_index.append(j[0])
                node_dest.append(i)



        node_source_list.append(np.int16(node_source))
        node_dest_list.append(np.int16(node_dest))
        edge_type_index_list.append(np.int8(edge_type_index))
    # 生成度向量
    for i in range(len(graph_edge)):
        _,x_unique = np.unique(node_dest_list[i], return_counts=True)



        node_source_decrease = np.array([x-1 for x in node_source_list[i]])
        dg_list.append(np.array(x_unique[node_source_decrease]))


    graph = [graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list,labels]
    return graph

def split(graph_vertix,graph_edge,y):
    rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=None)
    for train, test in rs.split(graph_vertix, graph_edge, y):
        X_train_vertix, X_test_vertix = np.array(graph_vertix)[train], np.array(graph_vertix)[test]
        X_train_edge, X_test_edge = np.array(graph_edge)[train], np.array(graph_edge)[test]
        y_train, y_test = y[train], y[test]


    pro_train = preprocess(X_train_vertix,X_train_edge,y_train)
    pro_test = preprocess(X_test_vertix,X_test_edge,y_test)
    return pro_train,pro_test


def train(graph_vertix, graph_edge, labels, T, epoch_num, batch_size, ln, s):
    model = OriLinearGNN(feat_dim=ln,  # 每个结点的特征向量
                         stat_dim=s,
                         T=T)
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    '''out_tansform = nn.Linear(in_features=s,
                             out_features=2,
                             bias=True)'''

    # train_set =  GraphDataset(pro_train)
    # trainloader = DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)



    for fold in range(5):
        train, test = split(graph_vertix, graph_edge, labels)
        LR = 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        for epoch in range(epoch_num):
            k = len(train[0]) // batch_size
            last_batch = len(train[0]) % batch_size
            num = 0
            for j in range(k):
                full_loss = 0
                for i in range(batch_size):
                    loss = 0
                    train_out = model(torch.from_numpy(train[0][i + j * batch_size]),
                                      torch.from_numpy(train[1][i + j * batch_size]),
                                      torch.from_numpy(train[2][i + j * batch_size]),
                                      torch.from_numpy(train[3][i + j * batch_size]),
                                      torch.from_numpy(train[4][i + j * batch_size]))

                    # train_out = train_out.view(-1,32)
                    # train_out = train_out.view(-1, 32)
                    # final_out = out_tansform(train_out)
                    target = (torch.from_numpy(train[5])).long()
                    # loss.view(-1)
                    loss = criterion(train_out, target[i].view(-1))
                    full_loss += loss
                    prediction = int(torch.max(F.softmax(train_out, dim=1), 1)[1])
                    if (prediction == train[5][i + j * batch_size]):
                        num += 1
                loss_a = full_loss / batch_size
                # Backward
                optimizer.zero_grad()
                loss_a.backward()
                optimizer.step()

                print('epoch %d, step %d, loss %.4f'
                      % (epoch + 1, j + 1, loss_a))
            #scheduler.step()
            print('训练集的acc为 %.4f' % (num / len(train[0])))

            correct = 0
            for num_test in range(len(test[0])):
                test_pred_out = model(torch.from_numpy(test[0][num_test]),
                                      torch.from_numpy(test[1][num_test]),
                                      torch.from_numpy(test[2][num_test]),
                                      torch.from_numpy(test[3][num_test]),
                                      torch.from_numpy(test[4][num_test]))
                # test_pred_out = test_pred_out.view(-1, 32)
                # final_pred_out = out_tansform(test_pred_out)
                target = (torch.from_numpy(test[5])).long()
                # loss.view(-1)
                prediction = int(torch.max(F.softmax(test_pred_out, dim=1), 1)[1])
                if (prediction == test[5][num_test]):
                    correct += 1
            #test_acc_list.append(correct / i)
            max_acc = 0.6
            if (correct / num_test) > max_acc:
                max_acc = correct / num_test
                print("save model")
                torch.save(model, 'gnn_model.pth')
            print('测试集的acc为 %.4f' % (correct / num_test))



if __name__ =='__main__':

  graph_vertix,graph_edge,labels = load_data(ln=100)
  #pro_train,pro_test = split(graph_vertix,graph_edge,labels)

  train(graph_vertix,graph_edge,labels,8,5,64,100,32)

