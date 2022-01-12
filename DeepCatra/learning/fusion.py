import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_reader import get_data
from gnn_preprocess import preprocess
from fusion_gpu import LSTM_net,sum_net
from hybrid_model import Hybrid_Network
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def get_dataset():


    labels, graph_vertix, graph_edge, lstm_feature = get_data('./dataset', 30)
    n = len(graph_vertix)
    graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list\
        = preprocess(graph_vertix, graph_edge)


    indices = np.random.permutation(len(graph_vertix))  # 将数据进行随机打乱

    graph_vertix = np.array(graph_vertix, dtype=object)[indices]
    node_source_list = np.array(node_source_list, dtype=object)[indices]
    node_dest_list = np.array(node_dest_list, dtype=object)[indices]
    edge_type_index_list = np.array(edge_type_index_list, dtype=object)[indices]
    dg_list = np.array(dg_list, dtype=object)[indices]
    lstm_feature = np.array(lstm_feature, dtype=object)[indices]

    labels = np.array(labels)[indices]

    # 取前百分之十为测试集，后百分之九十为训练集
    graph_vertix_test = graph_vertix[0:int(0.1*n)]
    graph_vertix_train = graph_vertix[int(0.1*n):]

    node_source_list_test = node_source_list[0:int(0.1 * n)]
    node_source_list_train = node_source_list[int(0.1 * n):]

    node_dest_list_test = node_dest_list[0:int(0.1 * n)]
    node_dest_list_train = node_dest_list[int(0.1 * n):]

    edge_type_index_list_test = edge_type_index_list[0:int(0.1 * n)]
    edge_type_index_list_train = edge_type_index_list[int(0.1 * n):]

    dg_list_test = dg_list[0:int(0.1 * n)]
    dg_list_train = dg_list[int(0.1 * n):]

    lstm_feature_test = lstm_feature[0:int(0.1 * n)]
    lstm_feature_train = lstm_feature[int(0.1 * n):]

    labels_test = labels[0:int(0.1 * n)]
    labels_train = labels[int(0.1 * n):]

    train = [graph_vertix_train, node_source_list_train, node_dest_list_train, edge_type_index_list_train, dg_list_train, lstm_feature_train,labels_train]
    test = [graph_vertix_test, node_source_list_test, node_dest_list_test, edge_type_index_list_test, dg_list_test, lstm_feature_test, labels_test]
    return train, test




def batch_iter(graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature, labels, batch_size):
    data_len = graph_vertix.shape[0]
    n_batch = int((data_len-1)/batch_size)+1
    #print("total batch: {}".format(n_batch))
    for i in range(n_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield labels[start_id:end_id], lstm_feature[start_id:end_id], \
              graph_vertix[start_id:end_id], node_source_list[start_id:end_id], \
              node_dest_list[start_id:end_id], edge_type_index_list[start_id:end_id], \
              dg_list[start_id:end_id]


def train(train,batch_size):

    epoch_num = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.tensor([])

    T = 8
    model = Hybrid_Network(30, 32, T)  # 加载模型
    model.to(device)

    Loss = nn.CrossEntropyLoss().to(device)          #定义损失
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)               #定义优化器
    best_val_acc = 0



    for epoch in range(epoch_num):
        step = 0
        correct = 0
        for labels, Lstm_feature, Graph_vertix, Node_source_list, Node_dest_list, Edge_type_index_list, Dg_list \
            in batch_iter(train[0], train[1], train[2], train[3], train[4], train[5], train[6], batch_size):

            #print(Lstm_feature.shape)
            #print(len(Lstm_feature))
            full_loss = 0
            for i in range(len(Graph_vertix)):
                lstm_feature = Lstm_feature[i].astype(int)
                graph_vertix = Graph_vertix[i].astype(int)
                node_source_list = Node_source_list[i].astype(int)
                node_dest_list = Node_dest_list[i].astype(int)
                edge_type_index_list = Edge_type_index_list[i].astype(int)
                dg_list = Dg_list[i].astype(int)


                lstm_feature = torch.LongTensor(lstm_feature)
                graph_vertix = torch.LongTensor(graph_vertix)
                node_source_list = torch.LongTensor(node_source_list)
                node_dest_list = torch.LongTensor(node_dest_list)
                edge_type_index_list = torch.LongTensor(edge_type_index_list)
                dg_list = torch.LongTensor(dg_list)

                length = len(lstm_feature)
                lstm_feature = torch.reshape(lstm_feature, (1, length, 100))

                out = model(graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature)

                labels = np.array(labels)
                labels = (torch.from_numpy(labels)).long()

                prediction = int(torch.max(F.softmax(out, dim=1), 1)[1])
                if (prediction == labels[i]):
                    correct += 1

                loss = Loss(out, labels[i].view(-1))
                #loss = Loss(out, labels[i])
                full_loss = full_loss+loss

            # Backward
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

            step += 1
            print('epoch %d, step %d, loss %.4f'
                  % (epoch + 1, step, full_loss))

        acc = correct/len(train[0])
        print("epoch :{} acc :{}".format(epoch + 1, acc))
        if acc > best_val_acc:
            torch.save(model.state_dict(), 'model_params.pkl')
            best_val_acc = acc







        # scheduler.step()
        #print('训练集的acc为 %.4f' % (num / len(train[0])))


def test(test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.tensor([])

    T = 8
    model = Hybrid_Network(30, 32, T)  # 加载模型
    model.to(device)
    model.load_state_dict(torch.load('model_params.pkl'))
    test_pred = []

    correct = 0
    for i in range(len(test[0])):
        lstm_feature = test[6][i].astype(int)
        graph_vertix = test[0][i].astype(int)
        node_source_list = test[1][i].astype(int)
        node_dest_list = test[0][2].astype(int)
        edge_type_index_list = test[3][i].astype(int)
        dg_list = test[4][i].astype(int)

        lstm_feature = torch.LongTensor(lstm_feature)
        graph_vertix = torch.LongTensor(graph_vertix)
        node_source_list = torch.LongTensor(node_source_list)
        node_dest_list = torch.LongTensor(node_dest_list)
        edge_type_index_list = torch.LongTensor(edge_type_index_list)
        dg_list = torch.LongTensor(dg_list)

        length = len(lstm_feature)
        lstm_feature = torch.reshape(lstm_feature, (1, length, 100))

        out = model(graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature)

        labels = np.array(train[5])
        labels = (torch.from_numpy(labels)).long()

        prediction = int(torch.max(F.softmax(out, dim=1), 1)[1])
        if (prediction == labels[i]):
            correct += 1
    print('测试集的准确率为：{}'.format(correct/len(test[0])))




    '''precision = precision_score(labels, test_pred, average='weighted')  # 输出精度
    recall = recall_score(labels, test_pred, average='weighted')  # 输出召回率
    print('precision: ', precision)
    print('recall: ', recall)'''

train_dataset, test_dataset = get_dataset()
train(train_dataset, 32)
test(test_dataset)






