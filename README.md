## PatrickStar: Parallel Training of Large Language Models via a Chunk-based Memory Management

![logo](./logo.png)

### Recent Progress
See [CHANGE_LOG.md](./CHANGE_LOG.md).

### Meeting PatrickStar
Pre-Trained Models (PTM) are becoming the hotspot of both NLP research and industry application. However, the training of PTMs requires enormous hardware resources, making it only accessible to a small portion of people in the AI community. Now, **PatrickStar will make PTM training available to everyone!**

Out-of-memory error (OOM) is the nightmare of every engineer training PTMs. We often have to introduce more GPUs to store the model params to prevent such errors. PatrickStar brings a better solution for such problem. With the **heterogeneous training** (DeepSpeed Zero Stage 3 also uses it), PatrickStar could fully use both the CPU and GPU memory so that you could use fewer GPUs to train larger models.

### System Design
The idea of Patrick is like this. The non-model data (mainly activations) varies during training, but the current heterogeneous training solutions are **statically** splitting the model data to CPU and GPU. To better use the GPU, PatrickStar proposes a **dynamic** memory scheduling with the help of a chunk-based memory management module. The memory management of PatrickStar supports offloading everything but the current computing part of the model to the CPU to save GPU. In addition, chunk-based memory management is efficient for collective communication when scaling to multiple GPUs.
See the paper and [this doc](./INSIDE.md) for the idea behind PatrickStar.

### Results
In experiment, Patrickstar v0.4.3 is able to train a **18 Billion**(18B) param model with 8xTesla V100 GPU and 240GB GPU memory in WeChat datacenter node, whose network topology is like [this](./doc/yard_network_fabric.md). PatrickStar is over twice as large as DeepSpeed. And the performance of PatrickStar is better for models of the same size as well. The pstar is PatrickStar v0.4.3. The deeps indicates performance of DeepSpeed v0.4.3 using the official example [DeepSpeed example](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/examples/ds_pretrain_gpt2-zero3.sh) zero3 stage with activation optimizations opening by default.

![alt perf](./doc/mgpu_scalability.png "performance testing result")

We also evaluated PatrickStar v0.4.3 on a single node of A100 SuperPod. It can train 68B model on 8xA100 with 1TB CPU memory, which is over 6x larger than DeepSpeed v0.5.7. Besides the model scale, PatrickStar is way more efficient than DeepSpeed. The benchmark scripts are in [here](./examples/benchmark).

![alt perf](./doc/one_node_perf_a100.png "performance testing result on SuperNode")

Detailed benchmark results on the WeChat AI data center and NVIDIA SuperPod are posted on this [Google Doc](https://docs.google.com/spreadsheets/d/136CWc_jA_2zC4h1r-6dzD4PrOvp6aw6uCDchEyQv6sE/edit?usp=sharing).


Scale PatrickStar to multiple machines (node) on SuperPod.
We succeed in training a GPT3-175B on 32 GPU. As far as we know, it is the first work
to run GPT3 on such a small GPU cluster.
Microsoft used 10,000 V100 to pertain GPT3.
Now you can finetune it or even pretrain your own one on 32 A100 GPU, amazing!

![alt perf](./doc/m_node_superpod.png "performance testing result on multiple Node of  SuperNode")


We've also trained the [CLUE-GPT2](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) model with PatrickStar, the loss and accuracy curve is shown below:

![CLUE-GPT2](./doc/clue-gpt2-loss-n-acc.png)

### Installation
```bash
pip install .
```

Note that PatrickStar requires gcc of version 7 or higher. You could also use NVIDIA NGC images, the following image is tested:

```bash
docker pull nvcr.io/nvidia/pytorch:21.06-py3
```

### Usage
PatrickStar is based on PyTorch, making it easy to migrate a pytorch project. Here is an example of PatrickStar:

```python
from patrickstar.runtime import initialize_engine

config = {
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0,
            "use_hybrid_adam": True,
        },
    },
    "fp16": {  # loss scaler params
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 2 ** 3,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "default_chunk_size": 64 * 1024 * 1024,
    "release_after_init": True,
    "use_cpu_embedding": False,
    "client": {
        "mem_tracer": {
            "use_async_mem_monitor": args.with_async_mem_monitor,
        }
    },
}

def model_func():
    # MyModel is a derived class for torch.nn.Module
    return MyModel(...)

model, optimizer = initialize_engine(model_func=model_func, local_rank=0, config=config)

...

for data in dataloader:
    optimizer.zero_grad()

    loss = model(data)
    model.backward(loss)
    optimizer.step()
```

We use the same `config` format as [DeepSpeed configuration JSON](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters), which mainly includes params of optimizer, loss scaler, and some PatrickStar-specific configuration.

For a detail explanation of the above example, please check the guide [here](./GUIDE.md)

For more examples, please check [here](./examples).

A quick-start benchmark script is [here](./examples/run_transformers.sh). It is executed with randomly generated data; therefore you do not need to prepare the real data. It also demonstrated all of the optimization techniques for patrickstar. For more optimization tricks running the benchmark see [Optimization Options](./doc/optimization_options.md).


### License
BSD 3-Clause License

### Cite Us
```
@article{fang2021patrickstar,
  title={PatrickStar: Parallel Training of Pre-trained Models via a Chunk-based Memory Management},
  author={Fang, Jiarui and Yu, Yang and Zhu, Zilin and Li, Shenggui and You, Yang and Zhou, Jie},
  journal={arXiv preprint arXiv:2108.05818},
  year={2021}
}
@article{fang2022parallel,
  title={Parallel Training of Pre-Trained Models via Chunk-Based Dynamic Memory Management},
  author={Fang, Jiarui and Zhu, Zilin and Li, Shenggui and Su, Hui and Yu, Yang and Zhou, Jie and You, Yang},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={34},
  number={1},
  pages={304--315},
  year={2022},
  publisher={IEEE}
}
```

### Contact Us
{jiaruifang, zilinzhu, josephyu}@tencent.com

Powered by WeChat AI Team, Tencent NLP Oteam.
