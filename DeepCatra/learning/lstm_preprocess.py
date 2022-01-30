import numpy as np

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
def split_opcode_seq(opcode_seq):
    opcode_seq_list=[]
    num=int(len(opcode_seq)/split_n)
    for i in range(0,num):
        opcode_seq_list.append(np.asarray(opcode_seq[len(opcode_seq)-(split_n*(i+1)):len(opcode_seq)-(split_n*i)]))  
    opcode_seq_list=reversed(opcode_seq_list)
    return opcode_seq_list
