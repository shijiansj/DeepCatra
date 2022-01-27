# GCN

This is the code of GCN.

## Install
Before running the code，you need first install the packages as follows:

PyG2.0.3

torch1.7.0

## Usage

run the commond:

python3 Generate_syscall_graph.py

to generate the syscall garaph of apk samples.A example is a benign sample with md5 hash value "00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B". Its syscall


sequence is in file "00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B_syscall.txt". And the generated syscall graph edges can be in the file 


"00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B_edge.txt".

run the commond:

python3 train.py

to train and evaluate the GCN model. The best model parameters has been saved in the file "model_params.pkl"

## References
T. S. John, et al. “Graph convolutional networks for android malware detection with system call graphs,” in ISEA-ISAP, 2020, pp. 162–170.

