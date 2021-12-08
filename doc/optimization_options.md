This page explains the optimization options for benchmarking.
Optimizations is divided into PatrickStar-related ones and general ones.
General Optimizations can be applied to any PyTorch-based frameworks.

## General Optimizations
1. Activation Checkpoing (a.k.a gradient checkpointing in [PyTorch](https://pytorch.org/docs/stable/checkpoint.html))
`--use_ckp`
Make sure this option is open for large model training. It can largely save activation memory footprint at cost of recomputing.

2. Activation Offloading
`--with_activation_offload`
Offload the checkpoints activation from GPU to CPU. Further Save GPU memory.
Note you have to use activation checkpoing first.

3. CPU Embedding
`--use_cpu_embedding`
nn.Embedding is conducted on CPU, save GPU memory. More importantly, it shrinks the chunk size. For some small model, the biggest layer is Embedding. Therefore, the chunk size has to larger than the embedding numel.

4. Tiling Linear (a.k.a Memory-centric tiling in [DeepSpeed](https://deepspeed.readthedocs.io/en/stable/zero3.html#memory-centric-tiling))
`--with_tiling_linear`
Memory-centric tiling (MCT) is able to split a param tensor of linear into pieces, and they do not need to be stored in contiguous memory space. This will help reduce chunk size. To achieve the best performance you have to tune the in_splits/out_splits of the parameters of the function.

## PatrickStar-related Optmizations

1. Memory Saving Communication.
`--with_mem_saving_com`
Use one-to-all communication to replace the original collective communication. More specifically, reduce scatter is replaced with Nx reduce. all gather is replaced with Nx bcast. In this way, we do not need to keep a Nx chunk buffer for distributed training, therefore saving the GPU memory. This method also changes the CPU-GPU and intra-GPU communication volume. In general, it reduces CPU-GPU comm volume at a cost of increasing intra-GPU bcast comm volume and also lower the intra-GPU bcast bandwidth. However, for some cases, it can improve the overall performance of the system from such tradeoff. It is suitable for training an extremely large model with a computing cluster with high-quality intra-GPU communication bandwidth, i.e. 50B model on a node of SuperPod. Details in Merge Request #250.

2. Memory Allocation Caching.
`--with_mem_cache`
Use a cache to allocate and release chunk memory. The cache is a size-limited queue, whose capacity is default as 2. It is helpful for Memory Saving Communication in distributed training. It avoid frequent release and allocate memory for remote chunks. See detail in #241.

2. Hybrid ADAM:
`--use_hybrid_adam`
Place Optimizer States (OS) on both CPU and GPU. Part of ADAM computation is conducted on CPU and the rest of computation is on GPU. On the contrary, Zero-Offload does ADAM on CPU only. This technique is able to accelerate ADAM computation for relative small model.

3. Activation Offload.
`--with_activation_offload`
Offload activation to CPU. Must used in combination with activation checkpointing (a.k.a gradient checkpoint in PyTorch).

4. Asyn Monitoring Memory with the Runtime Memory Tracer.
`--with_async_mem_monitor`
Async Sampling memory usage with an independent thread. It will bring a more accurate runtime
memory usage statistics. If you turn off this flag, memory usage sampling will triggered at the exact moment before or after operators (submodule in PyTorch) computing.


5. Static Partion.
`--with_static_partition`
PatirckStar is famous for dynamic partition model data. With help of this flag you can static partition model data between CPU and GPU. The max GPU used by chunks is `warmup_gpu_chunk_mem_ratio` * gpu_size. It is still better than Zero-Offload, which alway put all param and grad in GPU, to avoid OOM. It will lead to lower computing efficient than the default dynamic partition. But it is helpful to aggressively avoid OOM.

6. Release Remote Chunk After Initialization.
`release_after_init`
The is a computing efficient irrelevant option used for distributed training. It allocates memory for remote chunks but release it immediately. In this way, we can make sure the model parameter is randomly initialized the same as a serial version. Solve the problem with random seed. It is used in combination with the `--res_check` option to check the correctness of distributed training.

7. Adjusting the quota of CPU and GPU memory of memory tracer.
We did not expose this optimization as parameter passed through command line.
As shown in the pretrain_bert_demo.py, there is a json config for memory tracer setting. You can adjust the 4 ratio suffix values.

`warmup_gpu_chunk_mem_ratio`: the max gpu memory of a GPU can be used for chunks during the warmup iteration.

`overall_gpu_mem_ratio`: the available gpu mem size / real gpu mem capacity. Turn up the value if you meet cpu or gpu OOM during iteration.

`overall_cpu_mem_ratio`: the available cpu mem size / real cpu mem capacity. Turn up the value if you meet cpu or gpu OOM during iteration.

`margin_use_ratio`: Space to host optimizer states in GPU / the rest GPU space excluding the peak chunk-used space after warmup FWD+BWD.

`use_fake_dist`: a debug flag, to simulate multiple-GPU on one GPU. It is used when we are poor. After we have multi-GPU we deprecated this flag.

```
"mem_tracer": {
                    "use_async_mem_monitor": args.with_async_mem_monitor,
                    "warmup_gpu_chunk_mem_ratio": 0.1,
                    "overall_gpu_mem_ratio": 0.8,
                    "overall_cpu_mem_ratio": 0.8,
                    "margin_use_ratio": 0.8,
                    "use_fake_dist": False,
                    "with_static_partition": args.with_static_partition,
                },
```
