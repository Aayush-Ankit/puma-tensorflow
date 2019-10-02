# NOTE: slice bits are specified in low to high order slices
# NOTE: default CRS=4, ifmixed=True

hetero-2:
	CUDA_VISIBLE_DEVICES=0 python train_puma.py --logdir=asplos20_exps/puma_vgg16_slice_mixed_66665555 --slice_bits_list=6,6,6,6,5,5,5,5

hetero-3:
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --logdir=asplos20_exps/puma_vgg16_slice_mixed_66555544 --slice_bits_list=6,6,5,5,5,5,4,4
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --logdir=asplos20_exps/puma_vgg16_slice_mixed_55665544 --slice_bits_list=5,5,6,6,5,5,4,4
	CUDA_VISIBLE_DEVICES=1 python train_puma.py --logdir=asplos20_exps/puma_vgg16_slice_mixed_55664444 --slice_bits_list=5,5,6,6,4,4,4,4
