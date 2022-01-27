import os
import numpy as np
import networkx as nx


#读系统调用图的边
def edge_read(dataset_path):
    types = os.listdir(dataset_path)
    Edge = []
    label_list = []
    for type in types:
        type_path = dataset_path + '\\' + type
        apks_list = os.listdir(type_path)
        for i in range(len(apks_list)):
            if type == 'benign':
                label_list.append(np.array([0]))
            else:
                label_list.append(np.array([1]))
            edge_txt_path = type_path + '\\' + apks_list[i]
            edge_index = []
            edge = []
            edge_source = []
            edge_target = []
            with open(edge_txt_path, 'r') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    source_node = line.split(' ')[0]
                    des_node = line.split(' ')[1]
                    edge.append([int(source_node),int(des_node)])

            for i in range(len(edge)):
                edge_source.append(edge[i][0])
                edge_target.append(edge[i][1])

            edge_index.append(np.array(edge_source))
            edge_index.append(np.array(edge_target))

            edge_index = np.array(edge_index)
            Edge.append(edge_index)
    return Edge,label_list


#生成系统调用图中节点的标签
def create_node_feature(Edge):
    Node_Feature = []
    for i in range(len(Edge)):
        node_feature = []
        G = nx.DiGraph()
        edge = Edge[i]
        Node_katz = np.zeros(26, dtype=float)
        Node_betweeness = np.zeros(26, dtype=float)
        Node_closeness = np.zeros(26, dtype=float)
        Node_page_rank = np.zeros(26, dtype=float)
        for j in range(edge.shape[1]):
            G.add_edge(edge[0][j],edge[1][j])

        G_katz = nx.algorithms.centrality.katz_centrality_numpy(G, alpha=0.1, beta=1.0,  normalized=True, weight=None)
        G_closeness = nx.algorithms.centrality.closeness_centrality(G, u=None, distance=None, wf_improved=False)
        G_betweeness = nx.algorithms.centrality.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        G_page_rank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-08, nstart=None)

        for node in G_katz.keys():
            Node_katz[node-1] = G_katz[node]
        for node in G_betweeness.keys():
            Node_betweeness[node-1] = G_betweeness[node]
        for node in G_closeness.keys():
            Node_closeness[node-1] = G_closeness[node]
        for node in G_page_rank.keys():
            Node_page_rank[node-1] = G_page_rank[node]
        node_feature.append(Node_katz)
        node_feature.append(Node_betweeness)
        node_feature.append(Node_closeness)
        node_feature.append(Node_page_rank)
        node_feature = np.array(node_feature)
        node_feature = [ [row[i] for row in node_feature] for i in range(len(node_feature[0])) ]
        node_feature = np.array(node_feature)

        Node_Feature.append(node_feature)
    return Node_Feature
