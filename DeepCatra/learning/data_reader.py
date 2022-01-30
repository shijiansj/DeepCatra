import os
import numpy as np
from lstm_preprocess import encoding, split_opcode_seq


def load_my_data_split(deal_folder,split_length):

    opcode_dict=encoding()
    feature_data=[]
    with open(deal_folder, "r", encoding="utf-8") as file:

        opcode_seq = []
        for line in file.readlines():
            line = line.strip("\n")
            if line == "" and len(opcode_seq) != 0:
                feature_data.extend(split_opcode_seq(opcode_seq))
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

                    single_apk_data = load_my_data_split(opcode_path, split_length)
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
