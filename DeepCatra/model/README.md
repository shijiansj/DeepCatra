# Explanation

This is the explanation of the `model` directory.

## Install
Before running the code，you need first install the package as follows:

torch1.7.0

## Usage

run the commond:
```
python3 model_test.py apk_to_predict_path
```
to use the best model that saved in the file `model_best_params.pkl` to classify the apk. The parameter `apk_to_predict_path` is the directory of apk's DeepCatra feature.Note that
the  `apk_to_predict_path`  should contain `apk_name` subdirectory, and the subdirectory should contain three files:`edge.txt` ,`vertix.txt`,`sensitive_opcode_seq.txt`.
