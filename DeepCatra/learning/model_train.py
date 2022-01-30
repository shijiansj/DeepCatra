import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_reader import get_data
from gnn_preprocess import preprocess
from hybrid_model import Hybrid_Network
import datetime
import sys
from lstm_preprocess import encoding
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

opcode_dict = encoding()

def get_split_dataset(path,ln,split_length):

    labels, graph_vertix, graph_edge, lstm_feature = get_data(path, ln, split_length)
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

def batch_iter(graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list, lstm_feature, labels, batch_size):
    data_len = graph_vertix.shape[0]
    n_batch = int((data_len-1)/batch_size)+1
    for i in range(n_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield labels[start_id:end_id], lstm_feature[start_id:end_id], \
              graph_vertix[start_id:end_id], node_source_list[start_id:end_id], \
              node_dest_list[start_id:end_id], edge_type_index_list[start_id:end_id], \
              dg_list[start_id:end_id]


def train(train,valid_dataset,batch_size):

    epoch_num = 25

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = 10
    model = Hybrid_Network(13, 32, T)  # 加载模型
    model = model.to(device)
    Loss = nn.CrossEntropyLoss().to(device)          #定义损失
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)               #定义优化器
    best_f1= 0

    for epoch in range(epoch_num):
        step = 0
        correct = 0
        startTime2 = datetime.datetime.now()
        epoch_loss = 0
        model.train()
        for labels, Lstm_feature, Graph_vertix, Node_source_list, Node_dest_list, Edge_type_index_list, Dg_list \
            in batch_iter(train[0], train[1], train[2], train[3], train[4], train[5], train[6], batch_size):

            torch.cuda.empty_cache()
            full_loss = 0
            labels = torch.from_numpy(labels)
            labels = torch.LongTensor(labels)
            labels =labels.to(device)
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

                prediction = int(torch.max(out, 1)[1])
                if (prediction == labels[i]):
                    correct += 1

                loss = Loss(out, labels[i].view(-1))
                full_loss = full_loss+loss
            epoch_loss+=full_loss
            step += 1
            # Backward
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

        acc = correct/len(train[0])
        print('epoch %d, acc  %.4f' %(epoch + 1, acc))
        print('epoch %d, average loss  %.4f' % (epoch + 1, epoch_loss/len(train[6])))
        endTime2 = datetime.datetime.now()
        total_seconds = (endTime2 - startTime2).total_seconds()
        mins = total_seconds / 60
        print('epoch %d,所用时间为：%.2f' % (epoch + 1, mins))
        model_params_path = 'model_epoch'+str(epoch+1)+'_params.pkl'
        torch.save(model.state_dict(), model_params_path)
        f1 = valid(valid_dataset,model_params_path)
        if f1 > best_f1:
            torch.save(model.state_dict(), 'model_best_params.pkl')
            best_f1 = f1


def valid(test,model_params_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 10
    model = Hybrid_Network(13, 32, T)
    model.load_state_dict(torch.load(model_params_path))
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
        print('accuracy: ', accuracy)
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1-score: ', f1)
    return f1

def main():
    train_dataset_path = sys.argv[1]
    valid_dataset_path = sys.argv[2]
    traindataset = get_split_dataset(train_dataset_path, 13, 100)
    validdataset = get_split_dataset(valid_dataset_path, 13, 100)
    train(traindataset, validdataset, 16)

if __name__ == "__main__":
    main()


