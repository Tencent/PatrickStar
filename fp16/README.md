### FP16
DeepSpeed的优化必须采用fp16。因此Param data和grad都需要一份额外的FP16存储。
FP16和FP32的转化尽量在GPU上完成。

### 在PatrickStar存储
作为nn.Parameter的成员变量。
fp16_data, fp16_grad
fp32_data, fp32_data
背后都用chunked tensor存储。

### Forward
每层正向传播前需要将Param FP32 data (step在cpu计算出)转化成FP16 data。
在pre_forward_hook中加入GPU上的fp16_to_fp32计算。
1. access_data()，FP16 data ONHOLD->COMPUTE
2. foward()
3. fp16_to_fp32(FP16 data) -> FP32 data
4. release_data(), FP32 data COMPUTE->ONHOLD
5. release_fp16_data(), FP16 data COMPUTE->ONHOLD


### backward
每层反向传播将Param grad FP16转化为FP32

1. access_fp16_data(), FP16 data, ONHOLD->COMPUTE
2. access_fp16_grad(), FP16 grad, ONHOLD->COMPUTE
3. backward() -> FP32 grad (new ONHOLD)
4. remove_fp16_data(), FP16 data, COMPUTE->FREE
5. release_grad, fp32 grad COMPUTE->ONHOLD
