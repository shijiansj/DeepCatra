# Explanation

This is the explanation of the `learning` directory.

## Install
Before running the code，you need first install the package as follows:

torch1.7.0

## Usage

run the commond:
```
python3 model_train.py train_dataset_path valid_dataset_path
```
to train Hybrid model using train dataset and evaluate the model on the validation dataset. In the command, the parameter `train_dataset_path` is the path of train dataset and the parameter `valid_dataset_path` is the path of validation dataset. The best model parameters has been saved in the file `DeepCatra/DeepCatra/model/model_best_params.pkl`.
