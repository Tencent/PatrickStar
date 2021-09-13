# profiler 使用说明

请在对应位置插入如下代码来使用 profiler：

```python
from patrickstar.profiler import profiler

profiler.start()
# 这里是模型运行的代码
# ....
profiler.end()

profiler.save("file-to-save-profile-data.pkl")
```

可以用 `tools/profile_visualizer.py` 来可视化保存的 profile data：

```bash
python tools/profile_visualizer.py file-to-save-profile-data.pkl --memory_type=GPU
```

可以得到下图这样的结果：

![GPT3_8B model memory visualization](../../doc/profiler/GPT3_8B_memory.png)

其中红色曲线为模型实际使用的显存，蓝色为受 chunk 管理使用的显存。背景颜色中的绿、蓝、紫依次代表前向、反向、优化器更新 3 个训练阶段。
