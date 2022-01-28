# GCN

This is the code of GCN.

## Install
Before running the code，you need first install the packages as follows:

torch1.7.0

PyG2.0.3

## Usage

run the commond:
```
python3 Generate_syscall_graph.py syscall_datset_path edge_dataset_path
```
to generate the syscall garaph of apk samples.The parameter `syscall_datset_path` is the path of syscall sequence files of apk samples and the parameter `edge_datset_path` is
the path stores the generated edges of syscall graph. Note that the `syscall_datset_path` directory and the `edge_datset_path` should contain a `benign` subdirectory and a `malware` subdirectory. A example is a benign sample with md5 hash value `00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B`. Its syscall sequence is in file `00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B_syscall.txt`. And the generated syscall graph edges can be in the file 
`00B86FB1B89B3531CB5D0A40D23C40DFA3342B7010351D6F36C8F77596C1DA5B_edge.txt`.

then run the commond:

```
python3 model_train.py train_dataset_path valid_dataset_path
```
to train GCN model and evaluate the GCN model on validation dataset. The best model parameters has been saved in the file `model_best_params.pkl`.

finally run the command:
```
python3 model_test.py test_dataset_path
```
to evaluate the best model on the test dataset.

## References
T. S. John, et al. “Graph convolutional networks for android malware detection with system call graphs,” in ISEA-ISAP, 2020, pp. 162–170.

