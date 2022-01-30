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

def split_opcode_seq(opcode_seq,split_n):
    opcode_seq_list=[]
    num=int(len(opcode_seq)/split_n)
    diff_len = (num+1)*split_n - len(opcode_seq)
    arr = np.array(opcode_seq)
    opcode_seq = np.pad(arr, (0, diff_len), 'constant')
    for i in range(num+1):
        opcode_seq_list.append(opcode_seq[i*split_n:split_n*(i+1)])
    return opcode_seq_list