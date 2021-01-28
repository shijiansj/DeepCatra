import _pickle as cPickle
import os
import numpy as np
import importlib
import sys
import random
#import tensorflow as tf
from my_uncompress import *

#选择采用的策略：1：切割，2：填充，3：重复
strategy=2

#设置切割的每段长度
split_n=100
#设置选取的opcode序列数量,如果同时设为0，则会提取目录下所有的opcode序列
# opcode_seq_num=0
# test_opcode_seq_num=0#从CICI中选择的opcode序列个数： 12821

#设置填充的选择范围
opcode_max_length=100#选取opcode的长度上限
opcode_min_length=0#选取opcode的长度下限（不够则用0填充）

#当前提取的apk的良性与恶意标签，1表示良性，0表示恶意
benign_folder="C:/Users/73217/Desktop/研究生/lstm/data_my/benign_opcodes"#处理的良性apk特征目录
malicious_folder="C:/Users/73217/Desktop/研究生/lstm/data_my/malicious_opcodes"#处理的恶意apk特征目录
test_benign_folder="C:/Users/73217/Desktop/研究生/lstm/data_my/test_benign_opcodes"#用于存放测试的良性apk特征目录
test_malicious_folder="C:/Users/73217/Desktop/研究生/lstm/data_my/test_malicious_opcodes"#用于存放测试的恶意apk特征目录

# benign_folder="test_data/benign_opcodes"#处理的良性apk特征目录
# malicious_folder="test_data/malicious_opcodes"#处理的恶意apk特征目录
# test_benign_folder="test_data/test_benign_opcodes"#用于存放测试的良性apk特征目录
# test_malicious_folder="test_data/test_malicious_opcodes"#用于存放测试的恶意apk特征目录
test_folder="test_opcodes"

# apk_opcode_n=1#从每个apk中提取的opcode序列个数

def encoding():
    opcode_dict={}
    with open("opcodes_encoding.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip("\n")
            if line!="":
                opcode=line[:line.find(":")]
                num=int(line[line.find(":")+1:])
                opcode_dict[opcode]=num

    return opcode_dict

def padding(opcode_seq):
    diff_len = opcode_max_length - len(opcode_seq)
    #print("cha:",diff_len)

    if(diff_len>0):            #填充
        arr=np.array(opcode_seq)
        new_opcode_sequ=np.pad(arr,(0,diff_len),'constant')

        #print(type(arr))
        return np.asarray(new_opcode_sequ)
    else:                    #截断
        new_opcode_list = opcode_seq[-diff_len:]
        return np.asarray(new_opcode_list)

#对每个apk的opcode序列个数按n切割，不够的用0填充
# def deal_feature_data(feature_data):
#     new_feature_data=[]
#     for apks_feature_arr in feature_data:
#         if len(apks_feature_arr)>=apk_opcode_n:#apk特征序列少于20个，先剔除
#             temp_feature_data=[]
#             for i in range(0,len(apks_feature_arr),apk_opcode_n):
#                 temp_list=apks_feature_arr[i:i+apk_opcode_n]
#                 #print("转置前：",np.asarray(temp_list).shape)
#                 #convert_list=[[r[col] for r in temp_list] for col in range(len(temp_list[0]))]
#                 #print("转置后：",np.asarray(convert_list).shape)
#                 temp_feature_data.append(np.asarray(temp_list))
#             #每个apk对切割后的最后一个序列列表不够20的，直接剔除
#             if len(temp_feature_data[len(temp_feature_data)-1][0])<apk_opcode_n:
#                 temp_feature_data.pop()
#             new_feature_data.extend(np.asarray(temp_feature_data))
#
#     return new_feature_data

def load_my_data(deal_folder,flag):
    opcode_dict=encoding()

    feature_data=[]

    apks_opcode=os.listdir(deal_folder)
    random.shuffle(apks_opcode)  # 对数据集打乱
    for apk in apks_opcode:
        #print("apk:", apk)
        with open(deal_folder+"/"+apk,"r",encoding="utf-8") as file:
            opcode_seq=[]
            for line in file.readlines():
                line = line.strip("\n")
                if line=="" and len(opcode_seq)!=0:
                    #print("opcode_seq:",opcode_seq)
                    feature_data.append(padding(opcode_seq))
                    opcode_seq = []
                       # flag_data.append(np.str_(flag))

                elif line.find(":")==-1 and line!="":
                    opcode_seq.append(np.int32(opcode_dict[line]))

            if len(opcode_seq)!=0:
                feature_data.append(padding(opcode_seq))
                    #flag_data.append(np.str_(flag))
                #print("opcode_seq:", opcode_seq)


            #print("apks_feature:",apks_feature)
            # apks_feature_arr=np.asarray(apks_feature)
            #print("apks_feature维数：",apks_feature_arr.shape)

            #feature_data.append(apks_feature_arr)
            #print("opcode序列个数：",count)

            #输出apk中获取的每个20个序列
            # for apks_feature_arr in feature_data:
            #     print("apks_feature_arr:",apks_feature_arr)

    # 对所有apk中获取的opcode序列个数进行按n切割操作
    #feature_data = deal_feature_data(feature_data)
    # for i in feature_data:
    #     print("最终每个apk特征维数：",i.shape)  最终每个apk特征维数： (500, 20)
    #构建标签信息
    flag_data = []
    for i in range(0, len(feature_data)):
        flag_data.append(np.int32(flag))
    return feature_data, flag_data

def split_opcode_seq(opcode_seq):
   # print("原长度:",len(opcode_seq))
    opcode_seq_list=[]
    num=int(len(opcode_seq)/split_n)#110/20=5
    #print("num:",num)
    for i in range(0,num):
        opcode_seq_list.append(np.asarray(opcode_seq[len(opcode_seq)-(split_n*(i+1)):len(opcode_seq)-(split_n*i)]))    #？？？

    # print("切割后每个长度：",len(opcode_seq_list[0]))
    # print("切割后个数：",len(opcode_seq_list))
    # print("----------------------")
    opcode_seq_list=reversed(opcode_seq_list)
    return opcode_seq_list


def load_my_data_split(deal_folder, flag):

    opcode_dict = encoding()
    feature_data = []

    apks_opcode = os.listdir(deal_folder)
    random.shuffle(apks_opcode)#对数据集打乱

    for apk in apks_opcode:
        #print("apk:", apk)
        with open(deal_folder+"/"+apk, "r", encoding="utf-8") as file:

            opcode_seq=[]
            for line in file.readlines():
                line = line.strip("\n")
                if line=="" and len(opcode_seq)!=0:
                    #print("opcode_seq:",opcode_seq)
                    if len(opcode_seq)>=split_n:
                        feature_data.extend(split_opcode_seq(opcode_seq))#将一个opcode序列按n值切割成多个
                        # flag_data.append(np.str_(flag))
                    opcode_seq = []
                elif line.find(":") == -1 and line != "":
                    opcode_seq.append(np.int32(opcode_dict[line]))

            if len(opcode_seq)!=0:
                if len(opcode_seq) >= split_n:
                    feature_data.extend(split_opcode_seq(opcode_seq))  # 将一个opcode序列按n值切割成多个
                    #flag_data.append(np.str_(flag))
                #print("opcode_seq:", opcode_seq)

    #构建标签信息
    flag_data = []
    for i in range(0,len(feature_data)):
        flag_data.append(np.int32(flag))
    return feature_data, flag_data



def load_apk_data(deal_folder,flag):
    apks_feature_data=[]#三维

    opcode_dict = encoding()

    apks_opcode = os.listdir(deal_folder)
    for apk in apks_opcode:
        # print("apk:", apk)
        apk_feature_data = []
        with open(deal_folder + "/" + apk, "r", encoding="utf-8") as file:
            count = 0
            opcode_seq = []
            for line in file.readlines():
                line = line.strip("\n")
                if line == "" and len(opcode_seq) != 0:
                    # print("opcode_seq:",opcode_seq)
                    if len(opcode_seq) >= opcode_min_length and len(opcode_seq) <= opcode_max_length:
                        apk_feature_data.append(padding(opcode_seq))
                        # flag_data.append(np.str_(flag))
                        count += 1
                        opcode_seq = []
                elif line.find(":") == -1 and line != "":
                    opcode_seq.append(np.int32(opcode_dict[line]))

            if len(opcode_seq) != 0:
                if len(opcode_seq) >= opcode_min_length and len(opcode_seq) <= opcode_max_length:
                    apk_feature_data.append(padding(opcode_seq))
                    # flag_data.append(np.str_(flag))
                    # print("opcode_seq:", opcode_seq)
                    count += 1
        apks_feature_data.append(np.asarray(apk_feature_data))

    flag_data = []
    for i in range(0, len(apks_feature_data)):
        flag_data.append(np.int32(flag))
    return apks_feature_data, flag_data

def load_apk_data_split(deal_folder, flag):
    apks_feature_data=[]#三维

    opcode_dict = encoding()

    apks_opcode = os.listdir(deal_folder)
    for apk in apks_opcode:
        # print("apk:", apk)
        apk_feature_data = []
        with open(deal_folder + "/" + apk, "r", encoding="utf-8") as file:
            count = 0
            opcode_seq = []
            for line in file.readlines():
                line = line.strip("\n")
                if line == "" and len(opcode_seq) != 0:
                    # print("opcode_seq:",opcode_seq)
                    if len(opcode_seq)>=split_n:
                        apk_feature_data.extend(split_opcode_seq(opcode_seq))#将一个opcode序列按n值切割成多个
                        # flag_data.append(np.str_(flag))
                    count += 1
                    opcode_seq = []
                elif line.find(":") == -1 and line != "":
                    opcode_seq.append(np.int32(opcode_dict[line]))

            if len(opcode_seq) != 0:
                if len(opcode_seq) >= split_n:
                    apk_feature_data.extend(split_opcode_seq(opcode_seq))  # 将一个opcode序列按n值切割成多个
                    # flag_data.append(np.str_(flag))
                    # print("opcode_seq:", opcode_seq)
                    count += 1
        apks_feature_data.append(np.asarray(apk_feature_data))

    flag_data = []
    for i in range(0, len(apks_feature_data)):
        flag_data.append(np.int32(flag))
    return apks_feature_data, flag_data
    #return apks_feature_data

def get_train_data():
    feature_data=[]
    flag_data=[]
    if strategy==1:
        benign_feature_data, benign_flag_data=load_my_data_split(benign_folder,1)
        malicious_feature_data, malicious_flag_data = load_my_data_split(malicious_folder,0)
    elif strategy==2:
        benign_feature_data, benign_flag_data = load_my_data_split(benign_folder, 1)      #良性切割
        malicious_feature_data, malicious_flag_data = load_my_data(malicious_folder, 0)    #恶意填充
    else:
        benign_feature_data, benign_flag_data = load_my_data_split(benign_folder, 1)
        malicious_feature_data, malicious_flag_data = load_my_data_split(malicious_folder, 0)

    print("最终良性opcode序列个数：",len(benign_feature_data))
    print("最终恶意opcode序列个数：", len(malicious_feature_data))
    feature_data.extend(benign_feature_data)
    feature_data.extend(malicious_feature_data)
    flag_data.extend(benign_flag_data)
    flag_data.extend(malicious_flag_data)
    print("最终训练opcode序列数：",len(flag_data))
    # print("标签个数：",len(flag_data))
    # print("最终apk数据：", feature_data)
    # print("最终标签数据：", flag_data)
    return np.asarray(feature_data),np.asarray(flag_data)

def get_test_data():
    feature_data = []
    flag_data = []
    if strategy==1:
        test_benign_feature_data, benign_flag_data = load_my_data_split(test_benign_folder, 1)
        test_malicious_feature_data, malicious_flag_data = load_my_data_split(test_malicious_folder, 0)
    elif strategy==2:
        test_benign_feature_data, benign_flag_data = load_my_data_split(test_benign_folder, 1)
        test_malicious_feature_data, malicious_flag_data = load_my_data(test_malicious_folder, 0)
    else:
        test_benign_feature_data, benign_flag_data = load_my_data_split(test_benign_folder, 1)
        test_malicious_feature_data, malicious_flag_data = load_my_data_split(test_malicious_folder, 0)

    print("最终测试良性opcode序列个数：", len(test_benign_feature_data))
    print("最终测试恶意opcode序列个数：", len(test_malicious_feature_data))
    feature_data.extend(test_benign_feature_data)
    feature_data.extend(test_malicious_feature_data)
    flag_data.extend(benign_flag_data)
    flag_data.extend(malicious_flag_data)
    print("最终测试opcode序列数：", len(flag_data))#应该多于训练数组的1/10
    return np.asarray(feature_data), np.asarray(flag_data)

def get_test_apk_data():
    apks_feature_data = []
    apks_flag_data = []
    test_benign_feature_data, benign_flag_data = load_apk_data_split(test_benign_folder, 1)
    test_malicious_feature_data, malicious_flag_data = load_apk_data_split(test_malicious_folder, 0)
    apks_feature_data.extend(test_benign_feature_data)
    apks_feature_data.extend(test_malicious_feature_data)
    apks_flag_data.extend(benign_flag_data)
    apks_flag_data.extend(malicious_flag_data)
    print("最终apk数据个数：", len(apks_feature_data))
    return np.asarray(apks_feature_data), np.asarray(apks_flag_data)

def test_get_apk_feature():
    opcode_dict = encoding()
    # tf_opcode_dict=tf.convert_to_tensor(opcode_dict)
    feature_data, flag_data = load_my_data(test_folder, 1)
    arr=np.asarray(feature_data)
    # for opcode_seq in arr:
    #     print(opcode_seq.shape)
    #     print(tf.nn.embedding_lookup(tf_opcode_dict,opcode_seq).eval().shape)
    #
    #     break
    # print("arr维度：",arr.shape)
    # print("feature_data:",feature_data)
    # print("flag_data长度:",len(flag_data))
    # batch_x=arr[0:2]
    # print("batch_x1:",batch_x)#n*500
    # batch_x = opcode_seq_one_hot(batch_x, len(opcode_dict))
    # print("batch_x2:", batch_x)#n*500*190
    # print("一个opcode：",batch_x[0][1])
    return arr, np.asarray(flag_data)


if __name__ == '__main__':
    get_train_data()
    get_test_data()



    # x = [[1, 2], [3, 4], [5, 6]]
    # # random.shuffle(x)
    # print(x[:2])
    #test_get_apk_feature()
    # x=list(np.arange(20))
    # print(x)
    # for i in split_opcode_seq(x):
    #     print(i)

    #test_get_apk_feature()
    # opcode1 = [1, 2, 3]
    # opcode2 = [4, 5, 6]
    # opcode3 = [7, 8, 9]
    # lista=[]
    # lista.append(opcode1)
    # lista.append(opcode2)
    # lista.append(opcode3)
    # print(lista)
    # listb=[[r[col] for r in lista] for col in range(len(lista[0]))]
    # print(listb)

    # load_data2()
    # feature_data=deal_feature_data([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6]]])
    # print(feature_data)
    # opcode_seq1=[1,2,3,4]
    # opcode_seq2 = [1, 2, 3, 4]
    # temp=[]
    # arr1 = padding(opcode_seq1)
    # arr2 = padding(opcode_seq2)
    # print("arr:",type(arr1))
    # temp.append(arr1)
    # temp.append(arr2)
    # arr2=np.asarray(temp)
    # print("arr2:",type(arr2))
    # arr = np.asarray(opcode_seq)
    # print(type(arr))
    # print(arr)
    # #load_data2()
    # X, Y = load_data()
    # # print(X)
    # print(len(X))
    # print(len(Y))
    # print(Y)
    # print(X[0].shape)
