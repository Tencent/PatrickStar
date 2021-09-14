# profiler 使用说明

## 在代码中使用

请在对应位置插入如下代码来使用 profiler：

```python
from patrickstar.profiler import profiler

profiler.start()
# 这里是初始化以及模型运行的代码
# ...
model, optimizer = initialize_engine(...)
# ....
profiler.end()

profiler.save("file-to-save-profile-data.pkl")
```

## 可视化

可以用 `tools/profile_visualizer.py` 来可视化保存的 profile data。用下面的指令可以得到不同时刻的内存使用情况：

```bash
python tools/profile_visualizer.py file-to-save-profile-data.pkl --fig_type=memory --memory_type=GPU
```

![GPT3_8B model memory visualization](../../doc/profiler/GPT3_8B_memory.png)

其中红色曲线为模型实际使用的显存，蓝色为受 chunk 管理使用的显存。背景颜色中的绿、蓝、紫依次代表前向、反向、优化器更新 3 个训练阶段。

也可以用下面的指令可视化 chunk 在不同时刻的位置信息：

```bash
python tools/profile_visualizer.py file-to-save-profile-data.pkl --fig_type=access
```

![GPT3_8B model chunk location visualization on 4xV100](../../doc/profiler/GPT3_8B_4xV100_access.png)

其中红色部分为 chunk 在 CPU 上，蓝色部分为 chunk 在 GPU 上。从下向上 4 个部分分别表示 FP16_PARAM, FP32_PARAM, VARIANCE 和 MOMENTUM 类型的 chunk。
