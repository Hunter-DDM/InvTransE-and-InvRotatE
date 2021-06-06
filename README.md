# InvTransE & InvRotatE (ACL RepL4NLP Workshop 2021)

* Paper: Inductively Representing Out-of-Knowledge-Graph Entities by Optimal Estimation Under Translational Assumptions ([ArXiv version](https://arxiv.org/pdf/2009.12765.pdf), the final version will be updated soon)

## Requirements

* Pytorch (1.6.0)
* numpy (1.19.4)

## Training

1. Change the working directory to the root directory of this project. 
2. Run `bash runner/{method}/train_{dataset}.sh`.
   + `{method}` can be "InvTransE" or "InvRotatE". 
   + `{dataset}` specifies the dataset to evaluate, such as "WN11_OOKB_tail5000" and "FB15k_OOKB_tail10" (See all the 11 datasets in `data/`). 
   + The best configurations have been included in the shell scripts. Directly running these scripts can reproduce our results. 

## Evaluating

Tips: Before you evaluate a model on a dataset, please make sure that you have run the corresponding training script. 

1. Change the working directory to the root directory of this project. 
2. Run `bash runner/{method}/test_{dataset}.sh`.
   + `{method}` can be "InvTransE" or "InvRotatE". 
   + `{dataset}` specifies the dataset to evaluate, such as "WN11_OOKB_tail5000" and "FB15k_OOKB_tail10" (See all the 11 datasets in `data/`). 

## Citation

If you use this code for your research, please kindly cite our paper (The final citation format will be updated soon):

```
@article{dai2020inverse,
      author    = {Damai Dai and
                   Hua Zheng and
                   Fuli Luo and
                   Pengcheng Yang and
                   Baobao Chang and
                   Zhifang Sui},
      title     = {Inductively Representing Out-of-Knowledge-Graph Entities by Optimal
                   Estimation Under Translational Assumptions},
      journal   = {CoRR},
      volume    = {abs/2009.12765},
      year      = {2020},
      url       = {https://arxiv.org/abs/2009.12765}
    }
```

## Acknowledgements

We refer to [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) to develop our project. 

## Contact

Damai Dai: daidamai@pku.edu.cn
