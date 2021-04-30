### Client设计
client负责管理本进程的显存和内存使用。
系统中多个进程Client管理的显存和内存互相隔离的，这样避免了client之间进程进程间通信。

client可以register module，也可以register paramter。
通过param作为key来索引每个Parameter对应的data和grad。
Parameter的data和grad存储在以Chunk方式管理的tensor中(chunked tensor)。
Chunk是一段连续的内存空间，可以在内存或者显存中。Chunk可以在内存和显存间按需移动。

Parameter是client管理的最小单位。
client只能注册一个nn.Parameter，而无法单独注册一个tensor。
因为，一旦改变的tensor.data的位置，那么这个tensor也就不是原来的tensor了。
如果想让一个tensor被弄成chunked tensor形式，需要将tensor包装成nn.Parameter给PS注册。

我们为nn.Parameter增加成员变量来存储chunked tensor的指针。
目前ps_param_data和ps_param_grad是chunked tensor的data和grad对应数据的地址。
使用时将param.data和param.grad指向对应的chunked tensor。

我们还对梯度和参数的底层存储进行解耦。
值得注意的是，根据Pytorch nn.Paramter使用规则，Parameter的grad和data必须在同一个设备上。
一旦重置data tensor，那么grad自动被设置为None。
如果，重置grad tensor的设备与data tensor不一致则会报错。
我们自己定义的成员变量ps_param_data和ps_param_grad则可以存储在不同设备上。
从而分别独立管理data和grad的存储方式。

### Chunk 设计
Chunk底层一段连续的存储空间(如512 MB/1GB)，它被用来存储多个张量，但是其存储的张量只能有一种类型。
我们不能像在C++中使用`reinteprate_cast`方式来以特定数据类型访问Chunk的一个碎片。
在Pytorch中，每个torch.Tensor底层都是一个storage数据结构，
不能将一部分storage按照torch.Half访问，另一部分以torch.Float访问。
这就导致我们在不能同一个Chunk内同时存储FP16和FP32的数据。
必须释放FP32 chunk，新建一个FP16 chunk来完成FP32->FP16的转换。

通过Chunked tensor来存储参数和梯度，可以实现
1. 高速的数据传输。
一大块内存方式进行跨设备传输，可以充分利用CPU-GPU和GPU-GPU之间的带宽。
2. 重叠通信和计算。
传输chunk和其他chunk的计算可以重叠。
3. 内存复用，节省内存。
不同生命周期的tensor可以在一个chunk内复用。
通过这种方式可以避免grad FP32大小空间的分配。
4. 突破硬件存储空间限制。
我们不在需要CPU内存能够存储全部模型需要的参数，只需要CPU+GPU内存能够存储全部参数。

为了实现以上四点优势，我们对Chunk进行了一下几点设计。

##### Chunk Status
Chunk可以有三种状态，分别是COMPUTE，HOLD和FREE。
COMPUTE表示chunk在被用于计算中，此时应该放在计算设备上。
HOLD表示chunk不在计算中，但是Chunk内的内容必须被保存。
FREE表示chunk不在使用，内存可以被释放掉。

Chunk的状态是由它管理的Tensor决定的。
Tensor也有如上三种状态。
如果任何一个tensor是COMPUTE的，则Chunk状态是COMPUTE。
如果所有tensor是FREE的，则Chunk状态是FREE。
其他情况（即没有COMPUTE，有一个tensor是HOLD），Chunk状态是HOLD。

##### Lazy Allocation/Release
Tensor只有被需要计算时，它所在需要的Chunk内存才被分配出来。
当Tensor不再被需要时，它所在的Chunk内存被释放。

1. Instancely Release

在release tensor之后，检查其所在chunk的状态，如果变成FREE则释放内存。
这种方式有频繁分配释放内存的开销？
并不一定，tensor底层内存是PyTorch的CUDACachingMemory管理，并没有真正释放内存。

2. Cached Release

在release tensor之后，检查其所在chunk的状态，如果变成FREE并不释放内存。
留给下次分配时使用。
当已分配内存超过一定限制，再释放所有FREE状态的内存。

#####  Memory Reuse
首先，我们分析DNN训练时候的内存使用Pattern。
下面是一个FP16训练过程，使用client管理内存的流程。

1. 注册模型，param data FP16连续分配(A)。
2. 注册FP16 optimizer，将param data FP32连续分配。

step 0

3. 反向传播，pre-layer，param grad FP16 (B)连续分配，post-layer释放param data 16(A)
4. pre-step分配param grad FP32 (C)，释放grad fp16 (B)
5. 将M，V连续分配 (A,B可以free？)
6. post-step释放param grad FP32 (C)

step 1

7. 正向传播，pre-layer分配param data FP16 (A), at this moment
8. 反向传播，pre-layer分配param grad Fp16 (B)，post-layer释放param data 16(A)
9. pre-step分配分配param grad FP32(C)，释放grad fp16(B)
10. post-step释放param grad FP32 (C)

##### Tensor和Chunk的映射策略
我们可以设计一个Chunk和Tensor的最佳映射策略，来达到*节省内存*和*减少通信*的方案。
TODO
目前的方案是所有层的data和grad分配是连在一起的。

##### FP16 Optimizer
目前HybridPS支持apex的FP16 Optimier。

##### 效果
对弈个Simple Model。包含4层Linear，每个linear param data大小16，bias大小4。
参数总大小80个元素。

##### FP32训练
至少需要80 \*4B \*4(P+G+M+V)1280 B的GPU显存。
使用Chunked Tensor方式，
GPU显存最少需要显存的计算公式是：max(Chunk_size(Pgard_i) + Chunk_size(Pdata_i))为40*4=160B。
也就是反向传播一层需要的最大内存数目=Param+Grad=40 \*4B = 160B。
总体的内存需求不变，使用CPU内存最少为1280 - 160 = 1120B。

##### FP16训练
如果用apex的fp16 optimizer训练，
至少需要80 \*4B \*4(P32+G32+M32+V32) + 80 \* 2B \* 2(P16 + G16) = 1600 B的显存。
使用Chunked Tensor方式，
GPU仍然至少需要(2 chunk = 40 * 4B)160B显存。
使用CPU内存(P32 + M32 + V32 + G32) + 一个FP16 chunk(FP16-FP32转化过程需要一个额外chunk) 最少为320\*4 + 20\*2B = 1320B。
节省了几乎全部的P16 + G16的显存，大约是全部显存需求的20%。
总结，在FP16 optimier中使用chunked tensor，不仅可以保证最少的显存使用，还可以节省总体的内存需求。


## 性能优化
性能优化方式有二：

### 避免数据移动
一是，减少CPU-GPU移动的参数数量，
1. 设计一个聪明的chunk和tensor映射策略
#### 一、最佳映射
我们可以在预热阶段安排好chunk-tensor映射关系，之后的迭代根据tensor的唯一id来索引chunk。

##### 1. 布局(Layout)
Parameter的data和grad张量在chunk上的布局有两种方式。

A. 各自连续：
data0, data1, data2, ...
grad0, grad1, grad2, ...

B. 相互交错：
data0, grad0, data1, grad1, data2, grad2, ...

对于grad和data选择交错方案，因为BWD和adam计算时，data和grad都是成对出现。
这样坏处是FWD时候需要多分配一倍的内存。浪费内存，但是增加效率。

M，V肯定交错比较好，因为adam计算同时需要M，V，取一个chunk即可搞定。

##### 2. 对齐(Alignment)
data和grad
现在移动param的data和grad时候，最后一个chunk会带着M，V的数据
grad和data都在一个chunk，按照layer对齐。

设置合适的chunk尺寸，可以在预热过程统计参数的分配，设计动态可变的chunk尺寸，保证对齐。

##### 3. Adam计算位置
FP32：不管计算在CPU上还是GPU上，如果grad和data交错，M和V交错，那么取数据的通信量是一样的。
FP16: 如果adam计算在CPU上，FP16 data + Fp16 grad GPU->CPU, FP16 param CPU -> GPU
or FP32。 6x comm.

如果计算在gpu上，
M，V，P32应该都被挤到CPU上存放。
FP32 M, FP32 V, FP32 param, CPU->GPU, GPU->CPU. 24xComm.

所以默认还是在cpu上。

可以设计一个贪心策略。
检查，M，V，data，param的位置。如果它都在CPU，或者GPU则在对应设备上计算。如果它们分别CPU和GPU则在GPU上计算。

#### 二、重叠通信和计算

二是，重叠通信和计算，这需要异步的access, release接口。
可以把cpu分配为page-locked的pinned memory。
把所有的cpu都分配为pinned？似乎只有grad和param需要。

#### 三、加速Adam计算
三是，加速CPU的ADAM计算，让一部分在GPU上，另一部分在CPU上。
这个方法意义不大，因为对于BERT训练来说，我发现cpu adam时间占比跟小。



#### 撑大模型
在cuda分配抛出异常时候move out GPU内存？如何catch这个异常？
