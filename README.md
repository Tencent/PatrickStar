### PatrickStar: Break The Memory Wall of Parallel Training Large Language Models via Chunk-based Parameter Server

PatrickStar(派大星)是一种应用于大规模语言模型(LM)训练中的参数服务器。

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

PatrickStar的使用对象是Params, AccGrads, OS。它们都有全局唯一的性质。


#### 设计
PatrickStar是管理Param，AccGrad和OS的一种分布式存储模块，
它所分配的内存空间一个异构的设备的集合中，最典型的应用就是但机多卡GPU服务器，
设备集合包括一块CPU和多块GPU。

Chunk是内存管理的最小单位，它是一块连续的内存，可以坐落在CPU或者任何一块GPU之上。
GPU显存使用具有类似潮汐性质，在正向反向传播计算时，由于activation存在，GPU可供PatrickStar存储很少。
当Optimizer更新时，由于activation都释放掉了，GPU显存几乎可以全部供PatrickStar使用。
因为，为了更充分利用内存资源，增加计算通信重叠，Chunk机制营运而生。
当计算设备需要用一个Chunk的数据时，这个Chunk不在本地，则需要通过通信方式从远程设备上获取，
通信方式包括CPU-GPU之间拷贝，和GPU之间的collective通信。
[Chunk设计文档](./client/README.md)


PatrickStar由Manager和Client两部分组成。
[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)的数据并行方案中，
每一个进程负责一块GPU卡的计算。
基于此，PatrickStar在每一个GPU计算进程中一个Client。
它负责Param，AccGrad和OS的管理。
Manager是一个Singleton被所有计算进程共享的全局数据结构。

Client支持的方法
1. register_module(module)，将一个pytorch Module的Param (FP16)和Grad (FP16)存储由PatrickStar接管。
2. register_optim(optim)，讲一个优化器的底层空间(P FP32, G FP32, M FP32, V FP32)有PatrickStar接管。
3. access(param)，让param存储在计算设备上，param/grad计算设备是GPU，opti计算设备是cpu
4. release(param)，param不用了，可以被迁移了
4. allreduce/broadcast(local_tensor)，local_tensor每个进程拥有的本地tensor，allreduce结果是一个被Hybrid管理的张量。

Manager支持的方法
1. add/delete(dev_info, size)，dev分配/释放了size的空间
2. schedule(size, refer_device)，refer_device的进程需要申请size大小空间，返回一个可用的设备

#### 使用场景
ZeroDP stage1中，对param的分片进行bcast，allreduce，
这其实是HybridTensor是bcastFromMe,allreduceTooMe当底层设备是某个gpu的一种特例。
stage3的cpu-offload也是allreduceTooMe，bcastFromMe当底层设备是cpu的一种特例。

4个linear层，每层hidden dim = 40，数据个数1600+40 = 1640。
chunck size 2000.
硬件设置：
1.
*设置：
`manager.reset([4000 * 4] * 1, [32000 * 4 + 2 * 2000])`
*内存：
MA 101.0 KB         Max_MA 123.0 KB         CA 2048.0 KB         Max_CA 2048 KB
CPU Virtual Memory:  used = 3.91 GB, percent = 25.1%
*通信：
CPU-GPU data move elapse 0.0003352165222167969 sec, total elapse 0.1428055763244629 sec, total times 450, total amount 2516.0 KB
*时间：
is_ps True elapse 0.3383052349090576 sec

2.
*设置：
`manager.reset([40000 * 4] * 1, [32000 * 4 + 2 * 2000])`
*内存：
MA 144.0 KB         Max_MA 180.0 KB         CA 2048.0 KB         Max_CA 2048 KB
CPU Virtual Memory:  used = 3.9 GB, percent = 25.0%
total elapse 0.1428055763244629 sec, total times 450, total amount 2516.0 KB
*通信：
CPU-GPU data move elapse 0.0002455711364746094 sec, total elapse 0.03419327735900879 sec, total times 122, total amount 864.0 KB.
*时间：
is_ps True elapse 0.21094560623168945 sec

2.
*设置：
`manager.reset([40000 * 4] * 1, [32000 * 4 + 2 * 2000])`
梯度不设置为free，而是hold
*内存
MA 160.0 KB         Max_MA 192.0 KB         CA 2048.0 KB         Max_CA 2048 KB
CPU Virtual Memory:  used = 3.92 GB, percent = 25.1%
*通信
CPU-GPU data move elapse 0.0002434253692626953 sec, total elapse 0.033926963806152344 sec, total times 122, total amount 864.0 KB.
*时间
is_ps True elapse 0.20960783958435059

3. 不使用PatrickStar
*时间：
is_ps False elapse 0.07625579833984375
*内存:
MA 232.0 KB         Max_MA 245.5 KB         CA 2048.0 KB         Max_CA 2048 KB
CPU Virtual Memory:  used = 3.92 GB, percent = 25.1%

目前PatrickStar可以显著节省显存。但是内存用量没有变化，延迟显著增加。一部分是CPU-GPU数据移动引起的。另一部分可能是分配释放内存引起的。
Chunk复用，预取，通信计算重叠是性能优化的方向。
