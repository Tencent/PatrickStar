## Inside PatrickStar

The limited GPU memory is often the barrier toward training a large PTM model. Keeping the memory layout under scrutiny, we find there are 3 kinds of tensors in the GPU: parameter, gradient and optimizer state. Here, we use the Adam[1]optimizer, which is the most popular optimizer in PTM context. And for Adam, the optimizer state refers to the momentum and variance.

Researchers have proposed the mixed precision training[2], in which the model is trained with full precision float (fp32) and half precision float (fp16) combined. In this way, the parameter and gradient will be fp16 during forward and backward stage, which elimiates the memory consumption. And with the support of [tensor cores](https://www.nvidia.com/en-us/data-center/tensor-cores/), the fp16 computation are much faster than their fp32 counterparts. Therefore, we choose mixed precision training in PatrickStar. And to have better control of the memory, we implement our own training protocol instead of using the native `autocast`. Our protocol is similar to [apex](https://github.com/NVIDIA/apex) O2: we will cast the parameters to fp16 at the start of each iteration, do the forward and backward computation and cast the parameters and gradients back to fp32 for optimizer.

But for now, the GPU has to store all the tensors though only a small part of them are involved in the computation at the same time. It is rather tragic that the expensive, computation-specialized device are mainly used for storage... With this idea, ZeRO-Offload[3] proposed that we could move the Adam updation to CPU without losing much performance. In this way, we could move the memory-consuming momentum and variance (as they are both fp32 and the same size of parameters) to CPU and only save the parameters and gradients on GPU.

PatrickStar moves a step further! What if the total size of parameters is larger than the GPU memory? We noticed that most PTMs we are using today are in a linear structure (pack of transformer encoder layers). When calculating the latter layers, we do not need to store the former layers in GPU. And a natural idea emerges: why don't we offload the parameters that are not used for current computation to CPU?

However, directly offloading the parameters to CPU and loading them back to GPU when needed will harm the performance badly, as every computation will be waiting for serveral loading operation (CPU -> GPU). And because a single parameter are often small (the large model size are usually achieved by increasing the number of layers), this kind of memory movement could not fully utilize the PCIe bandwith between CPU and GPU. And for PyTorch (arguably the goto framework for NLP), the framework will cache the allocated GPU memorys so that the next allocation of the same size will be quicker. This results in memory fragmentation when moving back and forth with parameters of different sizes.

Here we introduce the chunk based memory management! We will organized the parameters into fixed size chunks (64MB is a usual size). And the memory loading and offloading are both done in unit of chunks. This will fully use the PCIe bandwidth and solve the memory fragmentation issue. Moreover, the parameters are inserted into chunks by their order of creation (which is also their order of usage in terms of PyTorch), when the computation requires the first parameter in a chunk, PatrickStar will move the chunk to GPU, which will also move subsequent parameters to the GPU. This makes the chunk-based memory management comes with prefetch. Hence, chunk-based memory management will maintain high performance even when the model size if much larger than the GPU memory.

The above is the gist of single GPU version of PatrickStar. Before talking about the distributed version, there are several points to clarify.

- To keep the performance of the model, we save a float32 copy of parameters on CPU to reduce truncation error;
- Because the size of the gradient is the same as that of its origin parameter, we will use the memory of the parameter to save the gradient when the parameter is no longer need during backward stage;
- We allow hybrid Adam: if there is enough GPU memory, we could move the chunk of optimizer states to GPU and do the Adam operations on GPU. This dynamic scheduling makes sure that PatrickStar also have execellent performance on small models.

For distributed training, ZeRO optimizer[4] splits the optimizer states across devices. Again, PatrickStar moves a step further! We consider spliting both the model and optimizer state across devices in units of chunk. The chunks are now classified as local chunk and remote chunk. The remote ones are not stored in the current process. When a parameter in a remote chunk is visited, all process will do a allgather operation to communicate the neighboring N chunks to each other (N is the number of processes or GPU devices). And when parameters in these N chunks are no longer needed, each parameter will release its fetched remote chunks. Each process only need to store the optimizer state and the fp32 copy of the parameters in local chunks. With the advantage of chunks, we achieve high throughput and prefetch in distributed model, while save the GPU memory to the extend of a model parallel algorithm (all process together will only have 1 copy of the model and optimizer state).

**NOTE**: This is a very simplified version of how PatrickStar works, please refer to our paper for detailed explanation and experiments. There are also a bunch of optimization not mentioned for simplicity.

### References

1. Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014.

2. Micikevicius P, Narang S, Alben J, et al. Mixed precision training[J]. arXiv preprint arXiv:1710.03740, 2017.

3. Jie Ren, Samyam Rajbhandari, et al. ZeRO-Offload: Democratizing Billion-Scale Model Training[J]. arXiv preprint arXiv:2101.06840, 2021.

4. Samyam Rajbhandari, Jeff Rasley, et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models[J]. arXiv preprint arXiv:1910.02054, 2019.
