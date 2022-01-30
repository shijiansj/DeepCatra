# Explanation

This is the explanation of the `learning` directory.

## Package dependency

Before running the code, please first install the package `torch1.7.0`.

## Usage

run the commond:
```
python3 model_train.py train_dataset_path valid_dataset_path
```
to train hybrid model using training dataset and evaluate the model on the validation dataset. In the command, `train_dataset_path` is the path of training dataset and `valid_dataset_path` is the path of validation dataset. The optimal model hyperparameters have been saved in the file `DeepCatra/DeepCatra/model/model_best_params.pkl`.
