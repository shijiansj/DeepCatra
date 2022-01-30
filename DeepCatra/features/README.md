# Description

This is the description of the `features` directory.

The DeepCatra feature of an apk sample includes three flies: `edge.txt`, `vertix.txt`, and `sensitive_opcode_seq.txt`. The first two files are the inputs of the GNN subnet, and the last file is the input of the BiLSTM subnet. The size of the features of all apk samples is more than 1GB. Therefore, we present an example here. The features for the complete dataset are available at [features](https://pan.baidu.com/s/1djWiVT91ypMdFmugR_1QjA), and the access code is `zrxp`.

The file `edge.txt` contains all the edges of an app's abstract flow graph, and in this file, every edge is saved in the format `edge_type source_vertix destination_vertix`. The file `vertix.txt` holds the vertex labels of an app's abstract flow graph. The `sensitive_opcode_seq.txt` holds the sensitive opcode sequence of an apk sample.
