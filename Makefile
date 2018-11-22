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

sim-puma-sigma:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --puma_sigma=0.00 --puma_alpha=0.00 --log_dir=vgg16_puma_sigma_0.00_alpha_0.00
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --puma_sigma=0.01 --puma_alpha=0.01 --log_dir=vgg16_puma_sigma_0.01_alpha_0.01
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --puma_sigma=0.10 --puma_alpha=0.01 --log_dir=vgg16_puma_sigma_0.10_alpha_0.01
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --puma_sigma=0.50 --puma_alpha=0.01 --log_dir=vgg16_puma_sigma_0.50_alpha_0.01

sim-puma-alpha:
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --log_dir=vgg16_puma_sigma_0.01_alpha_0.01 --puma_sigma=0.01 --puma_alpha=0.01
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --log_dir=vgg16_puma_sigma_0.01_alpha_0.10 --puma_sigma=0.01 --puma_alpha=0.10
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --log_dir=vgg16_puma_sigma_0.01_alpha_0.50 --puma_sigma=0.01 --puma_alpha=0.50
