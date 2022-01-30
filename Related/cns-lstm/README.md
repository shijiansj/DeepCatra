# LSTM

This is the code of LSTM.

## Install
Before running the code，you need first install the packages as follows:

tensorflow1.14.0

keras2.3.1

## Feature
To extract static features, run the command：
```
python get_apk_apis.py apks static_out
```
The parameter `apks` is the folder where the apk samples are stored, and the parameter `static_out` stores the extracted static features.

To extract dynamic features, first open the Android emulator, and then run the command：
```
python getSyscalls.py apks tmp dynamic_out
```
The parameter `apks` is the folder where the apk samples are stored, the parameter `tmp` is the folder where the intermediate results of the extraction process are stored, and the parameter `dynamic_out` stores the extracted dynamic features.

## Model
Due to size, all trained models are available at https://pan.baidu.com/s/1asC6C4QhhkgTkRnxfvaJzA , and the access code is `dvac`.
After getting the model, in order to make predictions for the apk of unknown class, run the command:
```
python LSTM_hybrid.py static_word2vec_7000.model dynamic_word2vec_10000.model Attn_bi_lstm_model_static_7000.h5 Attn_bi_lstm_model_dynamic_10000.h5 static_test dynamic_test
```
- static_word2vec_7000.model : word2vec model generated using static features
- dynamic_word2vec_10000.model : word2vec model generated using dynamic features
- Attn_bi_lstm_model_static_7000.h5 : The final model trained with static features after tuning with the validation set
- Attn_bi_lstm_model_dynamic_10000.h5 : The final model trained with dynamic features after tuning with the validation set
- static_test : Folder for storing apk static features
- dynamic_test : Folder for storing apk dynamic features

After executing this command, two files, lstm_pos_prob.txt and lstm_predict_labels.txt, will be generated to store the probability that the model predicts the apk to be malicious and the final prediction result respectively.

## References
D. Chaulagain, et al. “Hybrid analysis of android apps for security vetting using deep learning,” in CNS’20. IEEE, 2020, pp. 1–9.
