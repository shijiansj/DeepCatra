# Explanation

This is the explanation of the `model` directory.

## Package dependency

Before running the code, please first install the package `torch1.7.0`.

## Usage

run the commond:
```
python3 model_test.py apk_to_predict_path
```
to use the best model that saved in the file `model_best_params.pkl` to classify the apk sample. The parameter `apk_to_predict_path` is the directory of apk's DeepCatra feature. Note that `apk_to_predict_path` should contain `apk_name` subdirectory, and the subdirectory should hold three files, i.e., `edge.txt`, `vertix.txt`, and `sensitive_opcode_seq.txt`.
