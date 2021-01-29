import os
import numpy as np

def search_remove_null(path):
  files=os.listdir(path)   #查找路径下的所有的文件夹及文件
  num = 0
  num_null = 0
  for filee in  files:
      f=os.path.join(path,filee)   #使用绝对路径
      num +=1
      vande = os.listdir(f)
      for txt in vande:
          size = os.path.getsize(os.path.join(f,txt))
          if (size == 0) :
              os.remove(os.path.join(f,txt))
      if not os.listdir(f):
          os.rmdir(f)
          num_null +=1
  return num-num_null

def load_data(ln):
    path_benign = './dataset/benign'
    path_malicious = './dataset/malicious'
    num_benign = search_remove_null(str(path_benign))
    num_malicious = search_remove_null(str(path_malicious))

    path = "./dataset"  # 文件夹目录
    dir = os.listdir(path)  # 得到文件夹下的所有文件名称
    #print(dir)
    graph_vertix = []
    graph_edge = []
    labels = np.array([1 for i in range(num_benign + num_malicious)])


    i = -1

    for files in dir:  # 遍历文件夹
        sub_path = os.path.join(path,files)
        file = os.listdir(sub_path)
        i = i+1
        for apk in file:
            sub_sub_path = os.path.join(sub_path,apk)
            edge_path = os.path.join(sub_sub_path,'edge.txt')
            vertix_path = os.path.join(sub_sub_path,'vertix.txt')
            vandeando = os.listdir(sub_sub_path)
            for vande in vandeando:

                if(vande == 'edge.txt'):
                    edge_info = open(edge_path)
                    lines = edge_info.readlines()
                    edge = np.zeros((len(lines),3),dtype=int)
                    i=0

                    for line in lines:
                        curline = line.strip('\n')
                        curline = curline.split()
                        curline = [int(i) for i in curline]
                        curline = np.array(curline)
                        edge[i] = curline
                        i += 1
                    graph_edge.append(edge)


                if (vande == 'vertix.txt'):
                    vertix_info = open(vertix_path)
                    i=0
                    lines = vertix_info.readlines()
                    vertix = np.zeros((len(lines),ln),dtype=int)

                    for line in lines:
                        curline = line.strip('\n')
                        curline = curline.split()
                        curline = [int(i) for i in curline]

                        if(len(curline) < ln):
                            curline = list(curline + [0] * (ln - len(curline)))
                        if(len(curline) > ln):
                            curline = curline[-ln:]
                        curline = np.array(curline)
                        curline = curline.astype(float)
                        #归一化
                        curline -= np.mean(curline, axis=0)
                        curline /= np.std(curline, axis=0)
                        vertix[i] = curline
                        i += 1
                    graph_vertix.append(vertix)

    for j in range(num_benign):
        labels[j] = 0

    return graph_vertix,graph_edge,labels

