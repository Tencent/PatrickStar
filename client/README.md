### Client设计
client负责管理本进程的显存和内存使用。
系统中多个进程Client管理的内存时互相隔离的，这样避免了client之间进程进程间通信。

client可以register module，也可以register paramter。
通过param作为key来索引每个Parameter对应的data和grad。
Parameter的data和grad的内存通过Chunk来管理。

client无法单独管理一个tensor。
一旦改变的tensor.data的位置，那么这个tensor也就不是原来的tensor了。
原来tensor指针仍然指向原来的内存。
如果想让一个tensor被弄成chunked tensor形式，需要将tensor包装成nn.Parameter给PS注册。

每个param还要存储chunked tensor的指针。
目前ps_param_data和ps_param_grad是chunked tensor的data和grad对应数据的地址。
使用时将param.data和param.grad指向对应的chunked tensor。


#### register_module
将module所有param接管

#### register_param
接管param
_convert_to_ps_data->new_tensor
似乎这时不应该分配，只要记录shape即可

#### access_param
访问时按需new_tensor

#### relase_param


### Chunk的分配与释放

#### Lazy Allocation
新申请的tensor被分配在chunk的空间上，具体分配方式是lazy的。
正确逻辑
是否允许free chunk被cached？

1. free chunk立刻被释放
访问chunk之前把free chunk内存都释放（或者在release时候释放），访问的时候如果chunk不存在则分配。

2. free chunk被cache，只有内存申请爆表时候再释放。
在access需要

#### Memory Reuse
1. 注册模型，param data FP16连续分配(A)。
* 观察：分配了4个float16 chunk
2. 注册FP16 optimizer，将param data FP32连续分配。
* 观察：分配了4个float32 chunk
step 1
3. 反向传播，pre-layer，param grad FP16 (B)连续分配，post-layer释放param data 16(A)
* 观察：pre-layer分配1个FP16 chunk给grad，set data tensor to free，这里可以复用，只需要一个额外的chunk分配给第一个fp16 grad
4. pre-step分配param grad FP32 (C)，释放grad fp16 (B)
* 观察：在step之前，还有chunk在compute状态，没有被设置为hold
5. 将M，V连续分配 (A,B可以free？)
* 观察：FP16 optimizer和cpu_adam冲突了，exp_avg exp_avg_sq不会被分配
6. post-step释放param grad FP32 (C)
step 2
7. 正向传播，pre-layer分配param data FP16 (A), at this moment
8. 反向传播，pre-layer分配param grad Fp16 (B)，post-layer释放param data 16(A)
9. pre-step分配分配param grad FP32(C)，释放grad fp16(B)
10. post-step释放param grad FP32 (C)

In this way, A+B和C没有共同存在，因此可以复用同一块内存。

#### Locality
不需要显式地将data和grad分配在一起，局部性可以保证。
