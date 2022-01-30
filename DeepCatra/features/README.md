# Description

This is the description of the `features` directory.

The DeepCatra feature of a apk sample includes three flies: `edge.txt`,`vertix.txt`,`sensitive_opcode_seq.txt`. The former two files are the inputs of GNN child network, and the last file is the input of LSTM child network. The size of the features of all apk samples is more than 1GB. Therefore, we just upload an example here. The features of whole dataset are available at https://pan.baidu.com/s/1djWiVT91ypMdFmugR_1QjA , and the access code is `zrxp`.

The file `edge.txt` contains all the edges of a abstract graph of a apk sample, and in this file every edge is saved in this format `edge_type source_vertix destination_vertix`. The file `vertix.txt` saved the labels of the vertix of a abstract graph of a apk sample. The `sensitive_opcode_seq.txt` saved the sensitive opcode sequence of a apk sample.
