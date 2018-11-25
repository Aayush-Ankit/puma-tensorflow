target:
	@echo 'cifar100 fixedpoint training and evaluation'

train:
	CUDA_VISIBLE_DEVICES=3 python train.py --quant_delay=101 --log_dir=cifar100_train --restore=False

train-eval-fixed:
	#CUDA_VISIBLE_DEVICES=1 python train.py --quant_delay=101 --log_dir=cifar100_fixed --restore=False
	CUDA_VISIBLE_DEVICES=1 python test.py --quant_bits=16 --log_dir=cifar100_fixed_16bits --chpk_dir=cifar100_fixed
	CUDA_VISIBLE_DEVICES=1 python test.py --quant_bits=12 --log_dir=cifar100_fixed_12bits --chpk_dir=cifar100_fixed
	CUDA_VISIBLE_DEVICES=1 python test.py --quant_bits=8 --log_dir=cifar100_fixed_8bits --chpk_dir=cifar100_fixed
	CUDA_VISIBLE_DEVICES=1 python test.py --quant_bits=4 --log_dir=cifar100_fixed_4bits --chpk_dir=cifar100_fixed
	CUDA_VISIBLE_DEVICES=1 python test.py --quant_bits=2 --log_dir=cifar100_fixed_2bits --chpk_dir=cifar100_fixed

clean-train:
	rm -r cifar100_fixed

clean-eval:
	rm -r cifar100_fixed_*_bits

sim-new-nonideality1:
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.0 --puma_alpha=0.0 --logdir=puma_vgg16_sigma_0.0_alpha_0.0
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00010 --puma_alpha=0.00001 --logdir=puma_vgg16_sigma_0.00010_alpha_0.00001
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00001 --puma_alpha=0.00010 --logdir=puma_vgg16_sigma_0.00001_alpha_0.00010
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00001 --puma_alpha=0.00001 --logdir=puma_vgg16_sigma_0.00001_alpha_0.00001
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00010 --puma_alpha=0.00010 --logdir=puma_vgg16_sigma_0.00010_alpha_0.00010
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.1 --puma_alpha=0.1 --logdir=puma_vgg16_sigma_0.1_alpha_0.1

sim-new-nonideality2:
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.001 --puma_alpha=0.001 --logdir=puma_vgg16_sigma_0.001_alpha_0.001
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.001 --puma_alpha=0.010 --logdir=puma_vgg16_sigma_0.001_alpha_0.010
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.010 --puma_alpha=0.001 --logdir=puma_vgg16_sigma_0.010_alpha_0.001
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.01 --puma_alpha=0.01 --logdir=puma_vgg16_sigma_0.01_alpha_0.01
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.1 --puma_alpha=0.01 --logdir=puma_vgg16_sigma_0.1_alpha_0.01
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.01 --puma_alpha=0.1 --logdir=puma_vgg16_sigma_0.01_alpha_0.1
