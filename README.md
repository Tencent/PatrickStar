### PatrickStar: Break The Memory Wall of Parallel Training Large Language Models via Chunk-based Parameter Server

预训练模型是NLP近年来最大的算法创新。对于大的语言模型（Languagle Model, LM），单个GPU的显存无法存储训练所需的模型参数信息。这是传统数据并行技术在LM训练过程中无法胜任。为了解决这个问题，近年来出现了很多新的分布式训练解决方案，比如，模型并行，流水线并行和ZeroDP并行等。这些方法在一定程度上缓解了大模型和有限GPU显存之间的矛盾。但是，复现这些工作仍然需要如DGX-2类型的高性能计算设备。这些设备拥有目前最顶级GPU，CPU内存空间和GPU-GPU通信网络。移植这些方法到一些普通的计算设备时，这些解决方案的威力往往大打折扣。
为了在普通计算设备上，突破GPU显存对模型存储的限制，完成大LM的高效训练。本文提出了一个名为PatrickStart的训练解决方法。和已有方案不同，它采用一个以Chunk（大块）为内存分配和移动单位的参数服务，从而达在异构存储空间内灵活存储超大模型参数的效果。采用这种方案，可以以最小的存储空间需求，完成高效的大模型训练任务。


详细设计文档请移步(权限找jiaruifang)

https://docs.qq.com/doc/DYUVRZ3ljSmtIUWVK
