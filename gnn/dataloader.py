import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

def preprocess(graph_edge):
    # nodes_num_list = v_matrix.shape[0]

    # 现在需要生成两个向量
    # 第一个向量类似于
    #   [0, 0, 0, 1, 1, ..., 18, 18]
    # 其中的值表示节点的索引，连续相同索引的个数为该节点的度
    # 第二个向量类似于
    #   [1, 2, 4, 1, 4, ..., 11, 13]
    # 与第一个向量一一对应，表示第一个向量节点的邻居节点

    # 首先统计得到节点的度
    Degree_list = []  # type: List[List[Tuple[int, int]]]

    for i in range(len(graph_edge)):
        # degree_list = dict()
        degree_list = defaultdict(list)
        for j in range(graph_edge[i].shape[0]):
            # 反向边
            if (graph_edge[i][j][1] in degree_list):
                degree_list[graph_edge[i][j][1]].append((graph_edge[i][j][0] + 4, graph_edge[i][j][2]))
            else:
                # degree_list.append(n1)
                degree_list[graph_edge[i][j][1]] = [(graph_edge[i][j][0] + 4, graph_edge[i][j][2])]
            # 前向边
            if (graph_edge[i][j][2] in degree_list):
                degree_list[graph_edge[i][j][2]].append((graph_edge[i][j][0], graph_edge[i][j][1]))
            else:
                # degree_list.append(n2)
                degree_list[graph_edge[i][j][2]] = [(graph_edge[i][j][0], graph_edge[i][j][1])]

        Degree_list.append(degree_list)

    # 然后生成两个向量
    '''node_inds = []
    node_neis = []
    edge_type_index = []

    for n in node_list:
        node_inds += [node_dict["stoi"][n]] * len(Degree[n])
        node_neis += list(map(lambda x: node_dict["stoi"][x],list(Degree[n])))
        edge_type_index = '''

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
        '''node_source = np.array(node_source)
        node_source.astype(np.int64)
        edge_type_index = np.array(edge_type_index)
        edge_type_index.astype(np.int64)
        node_dest = np.array(node_dest)
        node_dest.astype(np.int64)'''

        node_source_list.append(np.int16(node_source))
        node_dest_list.append(np.int16(node_dest))
        edge_type_index_list.append(np.int8(edge_type_index))
    # 生成度向量
    for i in range(len(graph_edge)):
        _, x_unique = np.unique(node_dest_list[i], return_counts=True)
        '''print(x_unique.shape)
        print(node_dest_list[i].shape)
        print(node_dest_list[i])
        print(node_source_list[i])
        deg_list.append(x_unique)
        deg_list = np.array(deg_list)
        print(deg_list.shape)
        print(graph_edge[i].shape[0])
        print(node_source_list[i].shape)'''
        node_source_decrease = np.array([x - 1 for x in node_source_list[i]])
        dg_list.append(np.array(x_unique[node_source_decrease]))

    pro_dataset = [graph_vertix, node_source_list, node_dest_list, edge_type_index_list, dg_list]  # dglist

    return pro_dataset,labels


class GraphDataset(Dataset):
    def __init__(self, graphdataset):
        self.dataset = graphdataset[0]
        self.labels = graphdataset[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        graph_v = self.dataset[0][ind]
        node_source = self.dataset[1][ind]
        node_dest = self.dataset[2][ind]
        edge_index = self.dataset[3][ind]
        dg = self.dataset[4][ind]
        label = self.labels[ind]

        return graph_v,node_source,node_dest,edge_index,dg,label
        #return np.array(nodes, dtype=np.int64), np.array(edges, dtype=np.int64), np.array([label], dtype=np.int64)

