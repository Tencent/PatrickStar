### v0.4.4 Dec. 2021
The system is successfully evaluated on a multi-node system.
Add Memory-centric tiling from DeepSpeed. It trains a 18B model on WeChat Yard.
It supports 18B on on our WeChat Yard Platform.

### v0.4.3 Nov. 2021
PatrickStar is evaluated on A100 SuperPod.
Some optimizations are developed to futher improve the model scale and efficiency, include memory saving communication (MSC), allocation cache (CACHE).
A serious bug caused by asyn chunk copy using stream is identied and fixed.
It trains 50B model on a 8xA100 SuperPod node.


### v0.4.0 Nov. 2021,
With a better memory tracer, PatrickStar further improves the max model scale than v0.3.0 (15B vs 12B) on WeChat Yard Platform.

### v0.3.0 Oct. 2021.
Our initial version significantly surpasses DeepSpeed both in model-scale and computing efficiency.
