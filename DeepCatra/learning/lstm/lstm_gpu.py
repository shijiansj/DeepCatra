import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from my_data_reader6 import encoding, get_train_data
from my_uncompress import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

opcode_dict = encoding()
def get_data():
    trainbenign_flag = 0
    trainmalicious_flag = 0
    testbenign_flag = 0
    testmalicious_flag = 0
    data_X, data_Y = get_train_data()
    indices = np.random.permutation(data_X.shape[0])  # 将数据进行随机打乱
    data_X = data_X[indices]
    data_Y = data_Y[indices]
    test_X = data_X[0:int(0.1*data_X.shape[0])]        #取前百分之十为测试集，后百分之九十为训练集
    data_X = data_X[int(0.1*data_X.shape[0]):]
    test_Y = data_Y[0:int(0.1*data_Y.shape[0])]
    data_Y = data_Y[int(0.1*data_Y.shape[0]):]


    #data_Y = flag_one_hot(len(data_Y), data_Y)
    #test_Y = flag_one_hot(len(test_Y), test_Y)
    # for i in data_Y:
    #     if i == 1:
    #         trainbenign_flag += 1
    #     if i == 0:
    #         trainmalicious_flag += 1
    # print('训练的良性opcode序列数：{} 训练的恶意opcode序列数：{}'.format(trainbenign_flag, trainmalicious_flag))
    # for i in test_Y:
    #     if i == 1:
    #         testbenign_flag += 1
    #     if i == 0:
    #         testmalicious_flag += 1
    # print('测试的良性opcode序列数：{} 测试的恶意opcode序列数：{}'.format(testbenign_flag, testmalicious_flag))
    return data_X, data_Y, test_X, test_Y

def batch_iter(x, y, batch_size=128):
    data_len = x.shape[0]
    n_batch = int((data_len-1)/batch_size)+1
    print("total batch: {}".format(n_batch))
    for i in range(n_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id:end_id], y[start_id:end_id]

class LSTM_net(nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        self.embedding = nn.Embedding(len(opcode_dict),128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 2),
                                 nn.Softmax(dim=1))
    def forward(self, x):
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape)         #torch.Size([64, 100, 128])
        x,_ = self.lstm(x)
        #print(x.shape)         #torch.Size([64, 100, 512])
        x = self.fc1(x[:, -1, :])

        #x = self.fc1(x)
        #print(x.shape)         #torch.Size([64, 128])
        x = self.fc2(x)
        #print(x.shape)         #torch.Size([64, 2])
        return x

device = torch.device('cuda:0')

def train(data_X, data_Y):
    learning_rate = 0.001
    epochs = 2
    model = LSTM_net().to(device)                    #加载模型
    Loss = nn.CrossEntropyLoss().to(device)          #定义损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)               #定义优化器
    best_val_acc=0
    for epoch in range(epochs):
        batch_count=0
        for x_batch, y_batch in batch_iter(data_X, data_Y, batch_size=128):
            x = torch.LongTensor(x_batch)
            x = x.to(device)
            y = torch.LongTensor(y_batch)
            y = y.to(device)
            #print(x.shape)       torch.Size([64, 100])
            #print(y.shape)       torch.Size([64])
            out = model(x)
            loss = Loss(out, y)
            # print(out)
            # print(torch.max(out, 1)[1].numpy())
            #print(out.shape)      #torch.Size([64, 2])
            #print(y.shape)        #torch.Size([64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = np.mean((torch.argmax(out, 1) == y).cpu().numpy())
            batch_count += 1
            if batch_count % 20 == 0:         #每过20个batch输出一次结果
                print('epoch: {}, batch: {}, loss: {}, acc: {}'.format(epoch + 1, batch_count, loss.data, accuracy))
    torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
    #torch.save(model, 'lstm_model.pth')

        # if (epoch + 1) % 1 == 0:                                     # 对模型进行验证
        #     for x_batch, y_batch in batch_iter(val_X, val_Y, batch_size=64):
        #         x = torch.LongTensor(x_batch)
        #         y = torch.LongTensor(y_batch)
        #         out = model(x)
        #         loss = Loss(out, y)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         val_acc = np.mean((torch.argmax(out, 1) == y).numpy())
        #         if val_acc > best_val_acc:
        #             torch.save(model.state_dict(), 'model_params.pkl')
        #             best_val_acc = val_acc
        #         print("epoch :{} acc :{}".format(epoch+1, val_acc))

def test(test_X, test_Y):
    model = LSTM_net().to(device)
    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_pred = []
    #print(test_Y[0:64])
    #print(test_Y[64:128])
    with torch.no_grad():
        for x_batch, _ in batch_iter(test_X, test_Y, batch_size=128):
            test_X = torch.LongTensor(x_batch)
            test_X = test_X.to(device)
            out = model(test_X)
            pred_batch = torch.max(out, 1)[1].cpu().numpy()             #模型的预测结果
            #print(pred_batch)
            for i in range(len(pred_batch)):
                test_pred.append(pred_batch[i])
        test_pred = np.array(test_pred)
        #print(test_pred[0:64])
        #print(test_pred[64:128])
        precision = precision_score(test_Y, test_pred, average='weighted')  # 输出精度
        recall = recall_score(test_Y, test_pred, average='weighted')  # 输出召回率
        print('precision: ', precision)
        print('recall: ', recall)
if __name__=='__main__':
    data_X, data_Y, test_X, test_Y = get_data()  # 载入数据

    train(data_X, data_Y)        #训练并保存模型

    test(test_X, test_Y)         #输出测试结果
# list=[1,2,3,4]
# y=torch.Tensor(list)
# print(type(y))
# x=torch.zeros(1,1,3)
# print(x)
# print(torch.rand(1,3))
# print(torch.randn(3,1))
# print(torch.__version__)
# x = torch.randn((4,4), requires_grad=True)
# y = 2*x
# z = y.sum()
#
# print(z.requires_grad)  # True
#
# z.backward()
#
# print(x.grad)
# print(torch.cuda.is_available())
# print(numpy.__version__)