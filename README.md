# DeepCatra: Learning Flow- and Graph-based Behaviors for Android Malware Detection

This is the code and data repository of DeepCatra.

## Directory structure

- DeepCatra: The implementation and data of DeepCatra.
  - API_list: The critical API list in Java and smali.
  - features: All the opcode sequences and abstract flow graphs as features for embedding.
  - learning: The code for learning the deep neural network model of DeepCatra.
  - model: The detection model of DeepCatra (parameter tuned with validation set).
  - results: The evaluation results of metrics.
- Dataset: The dataset used in our evaluations.
- Related: Results of related work under comparison.
  - codaspy-cnn: results of [1].
  - cns-lstm: results of [2].
  - gcn: results of [3].

## References

[1] N. McLaughlin, et al. ¡°Deep android malware detection,¡± in CODASPY¡¯17. ACM, 2017, pp. 301¨C308.

[2] D. Chaulagain, et al. ¡°Hybrid analysis of android apps for security vetting using deep learning,¡± in CNS¡¯20. IEEE, 2020, pp. 1¨C9.

[3] T. S. John, et al. ¡°Graph convolutional networks for android malware detection with system call graphs,¡± in ISEA-ISAP, 2020, pp. 162¨C170.

## Contributor

- Jian Shi - School of Cyber-Engineering, Xidian University.
- Yafei Wu - School of Cyber-Engineering, Xidian University.
- Peicheng Wang - School of Cyber-Engineering, Xidian University.
- Cong Sun - School of Cyber-Engineering, Xidian University. CONTACT: suncong AT xidian DOT edu DOT cn

## License
[MIT](https://choosealicense.com/licenses/mit/)