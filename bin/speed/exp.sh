
CUDA_VISIBLE_DEVICES=MIG-be40e214-99f9-5863-a4e0-a402ab132d8e python mae_pretrain.py --reduction_factor=0.05 --seed=0 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.05-seed_0 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.05-seed_0 --scheduler=infinite_cosine

CUDA_VISIBLE_DEVICES=MIG-d541938e-5089-5142-b2c9-e5c2b4d6e063 python mae_pretrain.py --reduction_factor=0.05 --seed=1 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.05-seed_1 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.05-seed_1 --scheduler=infinite_cosine

CUDA_VISIBLE_DEVICES=MIG-5ed1e984-7e0c-515d-910c-40a999eb8ce5 python mae_pretrain.py --reduction_factor=0.05 --seed=2 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.05-seed_2 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.05-seed_2 --scheduler=infinite_cosine

CUDA_VISIBLE_DEVICES=MIG-835fc5ae-1bf5-54c5-840b-6c6d13895bc8 python mae_pretrain.py --reduction_factor=0.05 --seed=3 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.05-seed_3 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.05-seed_3 --scheduler=infinite_cosine


Next up

Running on GPU 0: reduction_factor=0.05, seed=4
CUDA_VISIBLE_DEVICES=MIG-be40e214-99f9-5863-a4e0-a402ab132d8e python mae_pretrain.py --reduction_factor=0.05 --seed=4 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.05-seed_4 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.05-seed_4 --scheduler=infinite_cosine

Running on GPU 0: reduction_factor=0.1, seed=3
CUDA_VISIBLE_DEVICES=MIG-be40e214-99f9-5863-a4e0-a402ab132d8e python mae_pretrain.py --reduction_factor=0.1 --seed=3 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.1-seed_3 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.1-seed_3 --scheduler=infinite_cosine

Running on GPU 0: reduction_factor=0.2, seed=2
CUDA_VISIBLE_DEVICES=MIG-be40e214-99f9-5863-a4e0-a402ab132d8e python mae_pretrain.py --reduction_factor=0.2 --seed=2 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.2-seed_2 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.2-seed_2 --scheduler=infinite_cosine


Running on GPU 1: reduction_factor=0.1, seed=0
CUDA_VISIBLE_DEVICES=MIG-d541938e-5089-5142-b2c9-e5c2b4d6e063 python mae_pretrain.py --reduction_factor=0.1 --seed=0 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.1-seed_0 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.1-seed_0 --scheduler=infinite_cosine

Running on GPU 1: reduction_factor=0.1, seed=4
CUDA_VISIBLE_DEVICES=MIG-d541938e-5089-5142-b2c9-e5c2b4d6e063 python mae_pretrain.py --reduction_factor=0.1 --seed=4 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.1-seed_4 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.1-seed_4 --scheduler=infinite_cosine

Running on GPU 1: reduction_factor=0.2, seed=3
CUDA_VISIBLE_DEVICES=MIG-d541938e-5089-5142-b2c9-e5c2b4d6e063 python mae_pretrain.py --reduction_factor=0.2 --seed=3 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.2-seed_3 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.2-seed_3 --scheduler=infinite_cosine


Running on GPU 2: reduction_factor=0.1, seed=1
CUDA_VISIBLE_DEVICES=MIG-5ed1e984-7e0c-515d-910c-40a999eb8ce5 python mae_pretrain.py --reduction_factor=0.1 --seed=1 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.1-seed_1 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.1-seed_1 --scheduler=infinite_cosine




Running on GPU 2: reduction_factor=0.2, seed=0
CUDA_VISIBLE_DEVICES=MIG-5ed1e984-7e0c-515d-910c-40a999eb8ce5 python mae_pretrain.py --reduction_factor=0.2 --seed=0 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.2-seed_0 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.2-seed_0 --scheduler=infinite_cosine

Running on GPU 3: reduction_factor=0.2, seed=1
CUDA_VISIBLE_DEVICES=MIG-835fc5ae-1bf5-54c5-840b-6c6d13895bc8 python mae_pretrain.py --reduction_factor=0.2 --seed=1 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.2-seed_1 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.2-seed_1 --scheduler=infinite_cosine

Running on GPU 3: reduction_factor=0.1, seed=2
CUDA_VISIBLE_DEVICES=MIG-835fc5ae-1bf5-54c5-840b-6c6d13895bc8 python mae_pretrain.py --reduction_factor=0.1 --seed=2 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.1-seed_2 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.1-seed_2 --scheduler=infinite_cosine

Running on GPU 2: reduction_factor=0.2, seed=4
CUDA_VISIBLE_DEVICES=MIG-5ed1e984-7e0c-515d-910c-40a999eb8ce5 python mae_pretrain.py --reduction_factor=0.2 --seed=4 --output_dir=output/nonsweep-infinite_cosine-reduction_factor_0.2-seed_4 --max_device_batch_size=512 --name=nonsweep-infinite_cosine-reduction_factor_0.2-seed_4 --scheduler=infinite_cosine
