# AudioCADAVAE

This is an environmental sound classification modification of the original PyTorch implementation of "Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders" (CVPR 2019).

Paper: https://arxiv.org/pdf/1812.01784.pdf
  
## Requirements
The code was implemented using Python 3.5.6 and the following packages:

```
torch==0.4.1
numpy==1.14.3
scipy==1.1.0
scikit_learn==0.20.3
networkx==1.11
```

Using Python 2 is not recommended.

## Data

Dataset are ESC-50 and FSC22. Audio embeddings from https://github.com/ysims/AudioEmbeddings. 

## Usage

Run `python experiment.py` with `--dataset` flag set to either `ESC-50` or `FSC22`. `--fold` is used to select the fold for evaluation, as defined in https://arxiv.org/abs/2412.03771. For ESC-50 they are `fold04`, `fold14`, `fold24`, `fold34` and `test`. For FSC22 they are `val` and `test`. 

## Citation

Cite the original paper using

```
@inproceedings{schonfeld2019generalized,
  title={Generalized zero-and few-shot learning via aligned variational autoencoders},
  author={Schonfeld, Edgar and Ebrahimi, Sayna and Sinha, Samarth and Darrell, Trevor and Akata, Zeynep},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8247--8255},
  year={2019}
}
```
