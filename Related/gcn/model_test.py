import torch
import sys
import numpy as np
from DataReader import create_node_feature,edge_read
from torch_geometric.data import DataLoader
import random
from torch_geometric.data import Data
from GCN_pyG import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


random.seed(1)

batch_size = 256


def generate_test_dataset(batch_size,test_dataset_path):
    Dataset_test = []
    Edge,Label_list = edge_read(test_dataset_path)
    Node_Feature = create_node_feature(Edge)

    all_data = list(zip(Edge, Node_Feature,Label_list))
    random.shuffle(all_data)
    Edge[:],Node_Feature[:],Label_list[:] = zip(*all_data)

    Edge_test = Edge[:]

    Node_Feature_test = Node_Feature[:]

    Label_list_test = Label_list[:]

    for i in range(len(Edge_test)):
        edge = torch.LongTensor(Edge_test[i])
        node_feature = torch.FloatTensor(Node_Feature_test[i])
        label = torch.LongTensor(Label_list_test[i])
        data = Data(x=node_feature,edge_index=edge,y=label)
        Dataset_test.append(data)

    testloader = DataLoader(Dataset_test, batch_size=batch_size, shuffle=True)

    return testloader



def test(loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load('model_best_params.pkl'))
    correct = 0
    y_pred = []
    y_true = []
    pos_prob = []
    with torch.no_grad():
        for data in loader:
            model.eval()
            data = data.to(device)
            output = model(data)
            prob = output.cpu().numpy()
            print(prob)
            y_pred.extend(np.argmax(prob, axis=1))  # 求每一行的最大值索引
            print(data.y.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
            print(prob[:,1])
            pos_prob.extend(prob[:,1])
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(data.y.view_as(pred)).sum().item()

    print(y_true)
    print(y_pred)
    print(pos_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')  # 输出精度
    recall = recall_score(y_true, y_pred, average='binary')  # 输出召回率
    f1 = f1_score(y_true, y_pred, average='binary')
    Confusion_matrix =confusion_matrix(y_true,y_pred)
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1-score: ', f1)
    print('confusion_matrix:\n', Confusion_matrix)

    fpr, tpr, thersholds = roc_curve(y_true, pos_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--',label='ROC (area = {0:.6f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    accuracy = correct/len(loader.dataset)
    print('Test Accuracy {:.6f}'.format( accuracy))




def main():
    test_dataset_path = sys.argv[1]
    testloader = generate_test_dataset(batch_size, test_dataset_path)
    test(testloader)

if __name__ == "__main__":
    main()


