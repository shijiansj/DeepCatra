import os
import numpy as np
from my_data_reader6 import encoding, split_opcode_seq
import pathlib
from collections import defaultdict



split_n=100



def search_remove_null(path):
  files=os.listdir(path)   #查找路径下的所有的文件夹及文件
  num = 0
  num_null = 0
  for filee in  files:
      f=os.path.join(path, filee)   #使用绝对路径
      num += 1
      vande = os.listdir(f)
      for txt in vande:
          size = os.path.getsize(os.path.join(f, txt))
          if (size == 0) :
              os.remove(os.path.join(f, txt))
      if not os.listdir(f):
          os.rmdir(f)
          num_null += 1
  return num-num_null
def search(dir):
    apks = os.listdir(dir)
    for file in apks:
        files = os.listdir(os.path.join(dir,file))
        if (len(files)<4):
            os.rmdir(os.path.join(dir,file))

def search_opcode(dir):
    apks = os.listdir(dir)
    for file in apks:
        opcode = file +'_sensitive_opcode_seq.txt'
        apk_path = os.path.join(dir, file)
        path = os.path.join(apk_path, opcode)
        size = os.path.getsize(path)
        if (size==0):
            print(file)

def load_my_data_split(deal_folder):

    opcode_dict=encoding()
    feature_data=[]
    with open(deal_folder, "r", encoding="utf-8") as file:

        opcode_seq = []
        for line in file.readlines():
            line = line.strip("\n")
            if line == "" and len(opcode_seq) != 0:
                # print("opcode_seq:",opcode_seq)
                #if len(opcode_seq) >= split_n:
                feature_data.extend(split_opcode_seq(opcode_seq))  # 将一个opcode序列按n值切割成多个
                    # flag_data.append(np.str_(flag))
                opcode_seq = []
            elif line.find(":") == -1 and line != "":
                opcode_seq.append(np.int32(opcode_dict[line]))

        if len(opcode_seq) != 0:
            if len(opcode_seq) >= split_n:
                feature_data.extend(split_opcode_seq(opcode_seq))  # 将一个opcode序列按n值切割成多个
    return feature_data

def get_data(path, ln):

    #把数据集集中空的去掉
    path_benign = './dataset/benign'
    path_malicious = './dataset/malware'
    num_benign = search_remove_null(str(path_benign))
    num_malware = search_remove_null(str(path_malicious))
    search_opcode(path_benign)

    #search(path_benign)

    #path = "./dataset"  # 测试集文件夹目录
    dir = os.listdir(path)  # 得到文件夹下的所有文件名称

    graph_vertix = []
    graph_edge = []
    lstm_feature = []
    opcode_dict = encoding()
    labels = [0 for i in range(num_malware + num_benign)]
    for x in range(num_malware):
        labels[num_benign + x] = 1

    i = -1
    for files in dir:  # 遍历文件夹
        sub_path = os.path.join(path, files)
        file = os.listdir(sub_path)
        i = i + 1
        for apk in file:
            sub_sub_path = os.path.join(sub_path, apk)
            edge_path = os.path.join(sub_sub_path, 'edge.txt')
            vertix_path = os.path.join(sub_sub_path, 'vertix.txt')
            opcode = apk + "_sensitive_opcode_seq.txt"
            opcode_path = os.path.join(sub_sub_path, opcode)
            vandeando = os.listdir(sub_sub_path)
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
                    j = 0
                    lines = vertix_info.readlines()
                    vertix = np.zeros((len(lines), ln), dtype=int)

                    for line in lines:
                        curline = line.strip('\n')
                        curline = curline.split()
                        curline = [int(i) for i in curline]

                        if (len(curline) < ln):
                            curline = list(curline + [0] * (ln - len(curline)))
                        if (len(curline) > ln):
                            curline = curline[-ln:]
                        curline = np.array(curline)
                        curline = curline.astype(float)
                        # 归一化
                        curline -= np.mean(curline, axis=0)
                        curline /= np.std(curline, axis=0)
                        vertix[j] = curline
                        j += 1
                    graph_vertix.append(np.array(vertix))
                if(vande == opcode):

                    single_apk_data = load_my_data_split(opcode_path)
                    single_apk_data = np.array(single_apk_data)
                    lstm_feature.append(np.array(single_apk_data))
    '''print(len(graph_edge))
    print(len(graph_vertix))
    print(len(labels))
    print(len(lstm_feature))'''
    return labels, graph_vertix, graph_edge, lstm_feature