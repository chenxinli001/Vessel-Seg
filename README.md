## Hierarchical Deep Network with Uncertainty aware Semi-supervised Learning for Vesse Segmentation (NCA 2021) ([Link](https://link.springer.com/article/10.1007/s00521-021-06578-3))


## Requirements

- Python == 3.7.4
- Pytorch == 1.1.0
- Torchvision ==0.3.0
- CUDA 8.0

## Datasets
- Retinal vessel segmentation
  - **AV-DRIVE**
  - **HRF**
  - **INSPIRE-AVR**
- Liver vessel segmentation
  - **IRCAD**

## Train 
* python `main.py`

## Evaualate on your own results
* working, will be open soon...

## Results on semi-supervised setting of retinal A/V segmentation
|  Methods  |  Acc   | Sp  | Sen|  F1|
|  ----  | ----  |  ----  | ----  |----  |
| Ours | 90.1 | 88.2 | 92.4 | 89.5|
| Supervised (Upper B) |91.6| 92.4|91.0|90.8|

## Results on supervised setting of liver vessel segmentation
|  Methods  | Vessel (Acc) | Portal Vessel (Acc) | Hepatic Vessel (Acc)|
|  ----  | ----  |  ----  | ----  |
|Merveille et al. | 90.0 | 98.0| 96.0 |
| Lebre et al. |97.0| 97.0|98.0|
| Ours|97.9| 98.9|99.7|

Please see more details in the paper

## Citation

If you find this repository useful, please cite our paper:
```
@article{li2022hierarchical,
  title={Hierarchical deep network with uncertainty-aware semi-supervised learning for vessel segmentation},
  author={Li, Chenxin and Ma, Wenao and Sun, Liyan and Ding, Xinghao and Huang, Yue and Wang, Guisheng and Yu, Yizhou},
  journal={Neural Computing and Applications},
  volume={34},
  number={4},
  pages={3151--3164},
  year={2022},
  publisher={Springer}
}
