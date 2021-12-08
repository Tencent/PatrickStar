### v0.4.4 Dec. 2021
The system is successfully evaluated on a multi-node system.
The benchmark scripts are integrated with memory-centric tiling borrowed from DeepSpeed.
It trains an 18B model on WeChat Yard.


### v0.4.3 Nov. 2021
The system is evaluated on A100 SuperPod.
Some optimizations are developed to improve further the model scale and efficiency, including memory saving communication (MSC) and allocation cache (CACHE).
A severe bug caused by asyn chunk copy using stream is identified and fixed.
It trains a 50B model on an 8xA100 SuperPod node.


### v0.4.0 Nov. 2021,
The system is upgraded with a better memory tracer.
We improve the max model scale further than v0.3.0 (15B vs. 12B) on the WeChat Yard Platform.

### v0.3.0 Oct. 2021.
Our initial version significantly surpasses DeepSpeed both in model-scale and computing efficiency.
