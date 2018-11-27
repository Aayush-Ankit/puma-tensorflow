target:
	@echo 'cifar100 fixedpoint training and evaluation'

# Non-ideality experiments
sim-nonideality-gpu0:
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.0 --puma_alpha=0.0 --logdir=puma_vgg16_sigma_0.0_alpha_0.0
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00010 --puma_alpha=0.00001 --logdir=puma_vgg16_sigma_0.00010_alpha_0.00001
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00001 --puma_alpha=0.00010 --logdir=puma_vgg16_sigma_0.00001_alpha_0.00010
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00001 --puma_alpha=0.00001 --logdir=puma_vgg16_sigma_0.00001_alpha_0.00001
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.00010 --puma_alpha=0.00010 --logdir=puma_vgg16_sigma_0.00010_alpha_0.00010
	CUDA_VISIBLE_DEVICES=0 python train_puma_nonideality.py --puma_sigma=0.1 --puma_alpha=0.1 --logdir=puma_vgg16_sigma_0.1_alpha_0.1

sim-nonideality-gpu2:
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.001 --puma_alpha=0.001 --logdir=puma_vgg16_sigma_0.001_alpha_0.001
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.001 --puma_alpha=0.010 --logdir=puma_vgg16_sigma_0.001_alpha_0.010
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.010 --puma_alpha=0.001 --logdir=puma_vgg16_sigma_0.010_alpha_0.001
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.01 --puma_alpha=0.01 --logdir=puma_vgg16_sigma_0.01_alpha_0.01
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.1 --puma_alpha=0.01 --logdir=puma_vgg16_sigma_0.1_alpha_0.01
	CUDA_VISIBLE_DEVICES=2 python train_puma_nonideality.py --puma_sigma=0.01 --puma_alpha=0.1 --logdir=puma_vgg16_sigma_0.01_alpha_0.1

# Outer product (slice_bits, crs freq) experiments
puma_train-crs1:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits --slice_bits=3 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits --slice_bits=4 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits --slice_bits=5 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits --slice_bits=6 --crs_freq=1

puma_train-crs16:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits --slice_bits=3 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits --slice_bits=4 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits --slice_bits=5 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits --slice_bits=6 --crs_freq=16

puma_train-crs256:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits --slice_bits=3 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits --slice_bits=4 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits --slice_bits=5 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits --slice_bits=6 --crs_freq=256

