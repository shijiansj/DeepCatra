# Explanation

This is the explanation of the `results` directory.

## Package dependency

Before running the code, please first install the package `torch1.7.0`.

## Usage

run the commond:
```
python3 model_test.py test_dataset_path
```
to evaluate the model in `DeepCatra/DeepCatra/model/model_best_params.pkl` on the testing dataset. The parameter `test_dataset_path` is the directory of the testing dataset. The prediction results are shown in the file `pred_result.txt`. There are three columns in the file：the md5 of apk sample, predicted probability for positive (malicious) class, the true labels of apk sample. Moreover, we evaluate the model with different metrics (including accuracy, precision, recall, F1, and AUC) with this python script. The values of different metrics of the model are in the file `DeepCatra_Result.json`.
