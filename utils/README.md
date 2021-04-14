---
marp: true
---

# Chunk设计
chunk底层一段连续的存储空间，它被用来存储多个张量。
鉴于，torch.Tensor有一个相同数据类型的的storage，
不能将一部分storage按照torch.Half访问，另一部分以torch.Float访问。
因此，chunk存储的张量只能有一种类型。
这就导致我们在同一个Chunk内同时存储FP16和FP32的数据。
必须释放FP32 chunk，新建一个FP16 chunk来完成FP32->FP16的转换。

优化Pytorch runtime内存的思路是，
在正反向传播计算时，把param的data和grad指向HybridPS管理的内存。

Pytorch nn.Paramter使用规则。
Parameter的grad和data必须在同一个设备上。
不能设置grad和data的device不同。
但是一旦重置data tensor，那么grad自动被设置为None。
但是重置grad tensor的设备与data tensor不一致则会报错。

对于FWD和BWD这没有影响，因为反向过程是先access_data到GPU，再access_grad到GPU。
但是，在grad FP16->FP32时候，单独设置了grad为cuda，此时data还在cpu上会报错。
