# Explanation

This is the explanation of the `results` directory.

## Install
Before running the code，you need first install the package as follows:

torch1.7.0

## Usage

run the commond:
```
python3 model_test.py test_dataset_path
```
to evaluate the model that saved in the directory `DeepCatra/DeepCatra/model/model_best_params.pkl` on the test dataset. The parameter `test_dataset_path` is the directory of DeepCatra test dataset. The prediction results are shown in the file `pred_result.txt`. There are three columns in the file：md5 hash value of apk samples in the test dataset, predicted probability for positive class(i.e. malicious type), the true labels of apk samples. Additionally, we evaluate the model with different metrics(including accuracy,precision,recall,F1 and AUC) in this python script. And the values of different metrics to the model have been saved in the file `DeepCatra_Result.json`.
