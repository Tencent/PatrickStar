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
2. 内存复用，节省内存。
不同生命周期的tensor可以在一个chunk内复用。
通过这种方式可以避免grad FP32大小空间的分配。
3. 突破硬件存储空间限制。
我们不在需要CPU内存能够存储全部模型需要的参数，只需要CPU+GPU内存能够存储全部参数。

为了实现以上三点优势，我们对Chunk进行了一下几点设计。

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

可见，由于hook的特定啊，模型参数的data和grad都是连续分配在一起，局部性可以自动保证。

##### FP16 Optimizer
TODO
设计了一个兼容HybridPS tensor的FP16方案。

#### 效果
对弈个Simple Model。包含4层Linear，每个linear param data大小16，bias大小4。
参数总大小80个元素。

##### FP32训练
至少需要80 *4B *4(P+G+M+V)1280 B的GPU显存。
使用Chunked Tensor方式，
GPU显存最少需要显存的计算公式是：max(Chunk_size(Pgard_i) + Chunk_size(Pdata_i))为40*4=160B。
也就是反向传播一层需要的最大内存数目=Param+Grad=40 *4B = 160B。
使用CPU内存最少为1280 - 160 = 1120B。

##### FP16训练
至少需要80 *4B *4(P32+G32+M32+V32) + 80 * 2B * 2(P16 + G16) = 1600 B的显存。
使用Chunked Tensor方式，
GPU仍然至少需要160B显存。
使用CPU内存最少为1600 - 160 = 1440B。
