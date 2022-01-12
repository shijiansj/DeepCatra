import numpy as np

# the 2d array of our samples,
# each component is a category label


def write_dict_4(data):
    all_list=data
    #print("list:",list)
    #一行表示对一个api编成独热码
    with open("one_hot.txt","w",encoding="utf-8") as file:
        for alllist in all_list:
            for list in alllist:
                for i in list:
                    # for j in i:
                    file.write(str(i[0])+" ")
                file.write("\n")

def opcode_seq_one_hot(opcode_seqs,one_hot_dim):
    opcode_seqs_list=[]
    for opcode_seq in opcode_seqs:
        result=np.zeros((opcode_seq.shape[0],one_hot_dim))
        for index in range(0,len(opcode_seq)):
            if opcode_seq[index]!=0:
                result[index][int(opcode_seq[index])-1]=1

        opcode_seqs_list.append(result)

    return np.asarray(opcode_seqs_list)

def flag_one_hot(batch_size, Y):
    B = np.zeros((batch_size, 2))
    B[np.arange(batch_size), Y] = 1
    # print(B)
    return B

if __name__=='__main__':
    # oopcode_seqs=[[1,2,3,4,0,0],[4,5,6,4,0,0]]
    # print(opcode_seq_one_hot(np.asarray(oopcode_seqs),8))
    print(np.arange(3))
    y=flag_one_hot(3,[1,0,1])
    print(y)
