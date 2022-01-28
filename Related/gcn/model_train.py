import torch
import torch.nn as nn
import sys
from DataReader import create_node_feature,edge_read
from torch_geometric.data import DataLoader
import random
from torch_geometric.data import Data
from GCN_pyG import Net

random.seed(1)

batch_size = 64



def generate_loader(batch_size,datasetPath):
    Dataset_test = []
    Edge,Label_list = edge_read(datasetPath)
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

    dataloader = DataLoader(Dataset_test, batch_size=batch_size, shuffle=True)

    return dataloader


def train(dataloader,loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Loss = nn.CrossEntropyLoss().to(device)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_nums = 100
    best_val_acc = 0
    for epoch in range(epoch_nums):
        loss_epoch=0
        step = 0
        correct = 0
        for data in dataloader:

            model.train()
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output, data.y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
            step+=1
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(data.y.view_as(pred)).sum().item()

        # loss_epoch += loss
        accuracy = correct / len(dataloader.dataset)
        print('Train Epoch: {}  Average Loss: {:.6f}  Accuracy:{:.4f}'.format(epoch+1, loss_epoch/len(dataloader),accuracy))

        correct_val = 0
        with torch.no_grad():
            for data in loader:
                model.eval()
                data = data.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct_val += pred.eq(data.y.view_as(pred)).sum().item()

        accuracy_val = correct_val / len(loader.dataset)
        print('Val Accuracy: ', accuracy_val)

        if accuracy_val > best_val_acc:
            torch.save(model.state_dict(), 'model_params.pkl')
            best_val_acc = accuracy


def main():
    train_dataset_path = sys.argv[1]
    valid_dataset_path = sys.argv[2]
    trainloader = generate_loader(batch_size, train_dataset_path)
    validloader = generate_loader(batch_size, valid_dataset_path)
    train(trainloader, validloader)

if __name__ == "__main__":
    main()
