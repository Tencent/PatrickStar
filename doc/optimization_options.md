This page explains the optimization options for benchmarking.
Optimizations are divided into PatrickStar-related ones and general ones.
General Optimizations can be applied to any PyTorch-based framework.

## General Optimizations
1. Activation Checkpoing (a.k.a gradient checkpointing in [PyTorch](https://pytorch.org/docs/stable/checkpoint.html))
`--use_ckp`
Make sure this option is open for large model training. It can primarily save activation memory footprint at the cost of recomputing.

1. Activation Offloading
`--with_activation_offload`
Offload the checkpoints activation from GPU to CPU. Further Save GPU memory.
Note you have to use activation checkpoing first.


1. Tiling Linear (a.k.a Memory-centric tiling in [DeepSpeed](https://deepspeed.readthedocs.io/en/stable/zero3.html#memory-centric-tiling))
`--with_tiling_linear`
Memory-centric tiling (MCT) can split a param tensor of linear into pieces, and they do not need to be stored in contiguous memory space. This will help reduce chunk size. However, to achieve the best performance, you have to tune the in_splits/out_splits of the function's parameters.

## PatrickStar-related Optmizations

1. Activation Offload.
`--with_activation_offload`
Offload activation to CPU. Must used in combination with activation checkpointing (a.k.a gradient checkpoint in PyTorch).

1. Asyn Monitoring Memory with the Runtime Memory Tracer.
`--with_async_mem_monitor`
Async Sampling memory usage with an independent thread. It will bring a more accurate runtime
memory usage statistics. If you turn off this flag, memory usage sampling will triggered at the exact moment before or after operators (submodule in PyTorch) computing.


1. Static Partion.
`--with_static_partition`
PatirckStar is famous for dynamic partition model data. With help of this flag you can static partition model data between CPU and GPU. The max GPU used by chunks is `warmup_gpu_chunk_mem_ratio` * gpu_size. It is still better than Zero-Offload, which alway put all param and grad in GPU, to avoid OOM. It will lead to lower computing efficient than the default dynamic partition. But it is helpful to aggressively avoid OOM.

1. Release Remote Chunk After Initialization.
`release_after_init`
The is a computing efficient irrelevant option used for distributed training. It allocates memory for remote chunks but release it immediately. In this way, we can make sure the model parameter is randomly initialized the same as a serial version. Solve the problem with random seed. It is used in combination with the `--res_check` option to check the correctness of distributed training.

1. Adjusting the quota of CPU and GPU memory of memory tracer.
We provide ways to adjust the CPU and GPU memory usage quota for the memory tracer. We did not expose this optimization as parameters passed through the command line. As shown in the pretrain_demo.py, there is a JSON config for the memory tracer setting. You can adjust the four ratio suffix values.

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
