import numpy as np
from collections import defaultdict

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
