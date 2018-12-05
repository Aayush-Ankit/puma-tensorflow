#Modelling PUMA computations in Tensorflow

###### Dataset path (download from here)
```
https://purdue0-my.sharepoint.com/:f:/g/personal/aankit_purdue_edu/Er-V3FCs07VMtDAgC96jTHMBItvjFrjKP8IWy76MmVPrNQ?e=oDWakf
```

### How to run?
```
    CUDA_VISIBLE_DEVICES=<specify gpu> python train.py --dataset=<specify path> --logdir=<specify path>
```

#### **NOTE: Current version runs only on one GPU**


### How to use tensorflow for GPU timing measurements ?
```
    RUN train.py, after copying Lines 158, 159, 164 from train_puma.py at similar places as in train_puma.py.
    Open Tensorboard, chose run-step on left pane, and chose compute time in checklist below.
```

