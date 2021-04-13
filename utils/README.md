### Chunk设计
chunk底层一段连续的存储空间，它被用来存储多个张量。
鉴于，torch.Tensor有一个相同数据类型的的storage，
不能将一部分storage按照torch.Half访问，另一部分以torch.Float访问。
因此，chunk存储的张量只能有一种类型。
这就导致我们在同一个Chunk内同时存储FP16和FP32的数据。
必须释放FP32 chunk，新建一个FP16 chunk来完成FP32->FP16的转换。
