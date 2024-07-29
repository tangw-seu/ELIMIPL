# A PyTorch Implementation of ELIMIPL

This is a PyTorch implementation of our paper "Exploiting Conjugate Label Information for Multi-Instance Partial-Label Learning", (**IJCAI'24**).

Authors: [Wei Tang](https://tangw-seu.github.io/), [Weijia Zhang](https://www.weijiazhangxh.com/), [Min-Ling Zhang](http://palm.seu.edu.cn/zhangml/)

```bib
@inproceedings{tang2024elimipl,
  author       = {Wei Tang and Weijia Zhang and Min-Ling Zhang},
  title        = {Exploiting Conjugate Label Information for Multi-Instance Partial-Label Learning},
  booktitle    = {Proceedings of the 33rd International Joint Conference on Artificial Intelligence, Jeju, South Korea},
  pages        = {1--11},
  year         = {2024},
}
```

If you are interested in multi-instance partial-label learning, the seminal work [MIPLGP](https://tangw-seu.github.io/publications/SCIS'23.pdf) and [DEMIPL](https://tangw-seu.github.io/publications/NeurIPS'23.pdf) may be helpful to you.

```bib
@article{tang2023miplgp,
  title        = {Multi-Instance Partial-Label Learning: {T}owards Exploiting Dual Inexact Supervision},
  author       = {Wei Tang and Weijia Zhang and Min-Ling Zhang},
  journal      = {Science China Information Sciences},
  volume       = {67},
  number       = {3},
  pages        = {Article 132103},
  year         = {2024},
}

@inproceedings{tang2023demipl,
  author       = {Wei Tang and Weijia Zhang and Min-Ling Zhang},
  title        = {Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning},
  booktitle    = {Advances in Neural Information Processing Systems 36, New Orleans, LA, USA},
  pages        = {56756--56771},
  year         = {2023},
}

```



## Requirements

```sh
numpy==1.21.5
scikit_learn==1.3.0
scipy==1.7.3
torch==1.12.0
```

To install the requirement packages, please run the following command:

```sh
pip install -r requirements.txt
```



## Demo

To reproduce the results of `MNIST-MIPL` dataset in the paper, please run the following command:

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --gamma 0.1 --mu 1. --L 128
```



## Parameter Settings

| Dataset             | learning rate | gamma | mu   |
| ------------------- | ------------- | ----- | ---- |
| MNIST_MIPL (r=1)    | 0.05          | 0.1   | 1    |
| MNIST_MIPL (r=2)    | 0.05          | 0.1   | 1    |
| MNIST_MIPL (r=3)    | 0.05          | 0.1   | 0.1  |
| FMNIST_MIPL (r=1)   | 0.01          | 0.5   | 1    |
| FMNIST_MIPL (r=2)   | 0.01          | 0.5   | 1    |
| FMNIST_MIPL (r=3)   | 0.05          | 0.5   | 1    |
| Birdsong_MIPL (r=1) | 0.01          | 10    | 10   |
| Birdsong_MIPL (r=2) | 0.01          | 10    | 10   |
| Birdsong_MIPL (r=3) | 0.01          | 10    | 10   |
| SIVAL_MIPL (r=1)    | 0.01          | 10    | 10   |
| SIVAL_MIPL (r=2)    | 0.01          | 10    | 10   |
| SIVAL_MIPL (r=3)    | 0.05          | 10    | 10   |
| CRC-MIPL-Row        | 0.01          | 10    | 10   |
| CRC-MIPL-SBN        | 0.01          | 10    | 10   |
| CRC-MIPL-KMeansSeg  | 0.01          | 10    | 10   |
| CRC-MIPL-SIFT       | 0.01          | 10    | 10   |



This package is only free for academic usage. Have fun!