### HybridPS

HybirdPS是一种应用于大规模语言模型(LM)训练中的参数服务器。

#### 问题背景
对于大的LM，单个GPU的显存无法存储训练所需的全部数据。
以ZeroDP为代表的解决方案，将LM训练所需的参数(Param)，梯度(Grad)，优化器状态(OS)存储在多卡GPU和CPU内存上。
为此，ZeroDP设计一套复杂的GPU-GPU通信和GPU-CPU offload方案。
ZeroDP是面向DGX型号高性能GPU集群开发的，这种设备配备32块V100，和4块拥有1.5TB的CPU内存。
面向普通的商用集群，在CPU内存和显卡规模受限情况下，ZeroDP的优化威力会大打折扣。
而问题的核心是，定制化的ZeroDP通信方式无法平衡内存和显存使用。往往导致CPU/GPU或者OOM情况发生。

在传统的参数服务器(PS)中，一份全局参数分布式存储在并行系统中，每个设备计算时获取一份参数副本。
这种方式尽可能少地引入通信开销，但是参数的重复导致内存资源的浪费，而内存时大LM训练的紧缺资源。
Hybrid区别于传统PS，让参数只在系统中存储一份，计算时按需移动参数的不同部分到相应的计算硬件上。

#### 使用对象
张量是深度学习的基本数据结构，LM的数据并行训练过程需要使用如下几种张量。

1. 模型参数(Params): 不同GPU使用相同的模型参数，全局唯一。
2. 梯度(Grads): 不同GPU的反向传播计算会产生不同的地图。
3. 规约梯度(AccGrads): 不同GPU的梯度规约后的结果，全局唯一。
3. 激活(Activations): 正反向计算的中间结果，不同GPU的计算过程产生不同的中间结果。
4. 优化器状态(Optimizer States, OS): Adam优化器需要的状态参数，包括momentum和variance，全局唯一。

HybridPS的使用对象是Params, AccGrads, OS。它们都有全局唯一的性质。


#### 设计
HybridPS是管理Param，AccGrad和OS的一种分布式存储模块，
它所分配的内存空间一个异构的设备的集合中，最典型的应用就是但机多卡GPU服务器，
设备集合包括一块CPU和多块GPU。

HybridPS由Manager和Client两部分组成。
[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)的数据并行方案中，
每一个进程负责一块GPU卡的计算。
基于此，HybridPS在每一个GPU计算进程中一个Client。
它负责Param，AccGrad和OS的管理。
Manager是一个Singleton被所有计算进程共享的全局数据结构。

Client支持的方法
1. register_tensor(tensor)，将一个pytorch tensor的底层存储接管到HybridPS中。
2. new_tensor(shape)，新申请一个空间被HybridPS管理的张量。
3. swap_out()，主动让贤，因为这个设备上需要分配非HybridPS管理的内存，HybridPS需要将自己管理的存储空间迁移到其他设备。
4. allreduce/broadcast(local_tensor)，local_tensor每个进程拥有的本地tensor，allreduce结果是一个被Hybrid管理的张量。

Manager支持的方法
1. add/remove(dev_info, size)，dev分配/释放了size的空间
2. schedule(size, refer_device)，refer_device的进程需要申请size大小空间，返回一个可用的设备


#### 使用场景
ZeroDP stage1中，对param的分片进行bcast，allreduce，
这其实是HybridTensor是bcastFromMe,allreduceTooMe当底层设备是某个gpu的一种特例。
stage3的cpu-offload也是allreduceTooMe，bcastFromMe当底层设备是cpu的一种特例。