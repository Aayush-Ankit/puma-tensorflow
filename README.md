
# PUMA Functional Simulator

The functional simulator models different components of matrix operations execution with ReRam backend.

## System requirements

| Requirement       | Version           |
| ----------------- | ----------------- |
| Anaconda          | 4.5.11            |
| Python            | 3.6.6 or higher   |         |
| Tensorflow-gpu    | tested on 1.10.0  |

## Quick Start (for installing tensorflow-gpu in conda)

```sh
conda create --name tf-gpu python=3.6
source activate tf-gpu
conda install -c anaconda tensorflow-gpu
```
Download ```CIFAR-100``` dataset and convert it to tfrecords files (train and test sets).

## Usage

Data Loader ```data_loader.py``` in repo has been tested for CIFAR-100 dataset, should be extensible to other datasets with changing the parameters such as number of classes, training and testing example etc.

The training and testing scripts use vgg16 models by default - ``vgg16_puma.py```.

Note: this version runs on 1 GPU only.

### Training

For training with PANTHER operations use ```ifpanther=True```.

```python train_puma.py --dataset=<my_path>```

### Testing

```python test.py --dataset=<my_path>```

### Using tensorboard

Tensorboard helps in viasualization of several statistics collected during the training run (accuracy, loss etc).

``` tensorboard --logdir=<my_name>:<my_logpath>

### Measuring runtime and memory consumption of operations in tensorboard

Uncomment ```Lines 185-187 in train_puma.pu``` - dumps metadata during training.
Launch tensorboard, choose run-step on left-pane, choose compute time or memory from the checklist.

### Authors

Aayush Ankit (Purdue University)




