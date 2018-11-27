# puma-tensorflow
Modelling PUMA computations in tensorflow

###### Dataset path (download from here) - https://purdue0-my.sharepoint.com/:f:/g/personal/aankit_purdue_edu/Er-V3FCs07VMtDAgC96jTHMBItvjFrjKP8IWy76MmVPrNQ?e=oDWakf

#### How to run: CUDA_VISIBLE_DEVICES=<specify gpu> python train.py --dataset=<specify path> --logdir=<specify path>
#### NOTE: Current version runs only on one GPU

#### How to use: For GPU only simulations (no puma modelling) - use train.py; For puma simulations (slice, quantize, saturate and or non-ideality) - use train_puma.py
