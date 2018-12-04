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

# Outer product (slice_bits, crs freq) experiments - (num_slices=8, default_storage = 2bits per device, batch_size = 64)
puma_train-crs1:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits --slice_bits=3 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits --slice_bits=4 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits --slice_bits=5 --crs_freq=1
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits --slice_bits=6 --crs_freq=1

puma_train-crs4:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_4 --slice_bits=3 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_4 --slice_bits=4 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_4 --slice_bits=5 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_4 --slice_bits=6 --crs_freq=4

puma_train-crs16:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_16 --slice_bits=3 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_16 --slice_bits=4 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_16 --slice_bits=5 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_16 --slice_bits=6 --crs_freq=16

puma_train-crs32:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_32 --slice_bits=3 --crs_freq=32
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_32 --slice_bits=4 --crs_freq=32
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_32 --slice_bits=5 --crs_freq=32
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_32 --slice_bits=6 --crs_freq=32

puma_train-crs64:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_64 --slice_bits=3 --crs_freq=64
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_64 --slice_bits=4 --crs_freq=64
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_64 --slice_bits=5 --crs_freq=64
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_64 --slice_bits=6 --crs_freq=64

puma_train-crs256:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_256 --slice_bits=3 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_256 --slice_bits=4 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_256 --slice_bits=5 --crs_freq=256
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_256 --slice_bits=6 --crs_freq=256

puma_train-crs782:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_3bits_crsfreq_782 --slice_bits=3 --crs_freq=782
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_4bits_crsfreq_782 --slice_bits=4 --crs_freq=782
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_5bits_crsfreq_782 --slice_bits=5 --crs_freq=782
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_6bits_crsfreq_782 --slice_bits=6 --crs_freq=782

puma_mixedprec-crs16:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66664444_crsfreq_16 --ifmixed=True --slice_bits_list=6,6,6,6,4,4,4,4 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55554444_crsfreq_16 --ifmixed=True --slice_bits_list=5,5,5,5,4,4,4,4 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55664433_crsfreq_16 --ifmixed=True --slice_bits_list=5,5,6,6,4,4,3,3 --crs_freq=16
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66663333_crsfreq_16 --ifmixed=True --slice_bits_list=6,6,6,6,3,3,3,3 --crs_freq=16

puma_mixedprec-crs4:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66664444_crsfreq_4 --ifmixed=True --slice_bits_list=6,6,6,6,4,4,4,4 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55554444_crsfreq_4 --ifmixed=True --slice_bits_list=5,5,5,5,4,4,4,4 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55664433_crsfreq_4 --ifmixed=True --slice_bits_list=5,5,6,6,4,4,3,3 --crs_freq=4
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66663333_crsfreq_4 --ifmixed=True --slice_bits_list=6,6,6,6,3,3,3,3 --crs_freq=4

puma_mixedprec-crs6:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66664444_crsfreq_6 --ifmixed=True --slice_bits_list=6,6,6,6,4,4,4,4 --crs_freq=6
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55554444_crsfreq_6 --ifmixed=True --slice_bits_list=5,5,5,5,4,4,4,4 --crs_freq=6
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55664433_crsfreq_6 --ifmixed=True --slice_bits_list=5,5,6,6,4,4,3,3 --crs_freq=6
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66663333_crsfreq_6 --ifmixed=True --slice_bits_list=6,6,6,6,3,3,3,3 --crs_freq=6

puma_mixedprec-crs8:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66664444_crsfreq_8 --ifmixed=True --slice_bits_list=6,6,6,6,4,4,4,4 --crs_freq=8
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55554444_crsfreq_8 --ifmixed=True --slice_bits_list=5,5,5,5,4,4,4,4 --crs_freq=8
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_55664433_crsfreq_8 --ifmixed=True --slice_bits_list=5,5,6,6,4,4,3,3 --crs_freq=8
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=puma_vgg16_slice_mixed_66663333_crsfreq_8 --ifmixed=True --slice_bits_list=6,6,6,6,3,3,3,3 --crs_freq=8
