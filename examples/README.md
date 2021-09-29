## 派大星使用实例

### 在 huggingface 中使用派大星

`huggingface_bert.py` 是一个用派大星 fine-tune huggingface 模型的例子。可以通过对比这个例子和 [huggingface 官方 fine-tune 的例子](https://huggingface.co/transformers/custom_datasets.html#seq-imdb) 来了解如何向已有的项目中引入派大星。

在运行该例子之前，首先需要准备数据：

```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

并修改 `get_dataset()` 为指定的目录。之后，直接运行即可：

```bash
python huggingface_bert.py
```

### 利用派大星训练大规模模型

`run_bert.sh` 和 `pretrain_bert_demo.py` 是用派大星训练大规模预训练模型的例子。可以通过配置 `run_bert.sh` 中的参数来设置不同大小的模型。

使用 Bert 模型检查收敛精度。

```bash
bash RES_CHECK=1 run_bert.sh
```

可以通过下面的设置来调整运行模型的规模，此时我们关闭精度检查：

```bash
env MODEL_NAME=GPT2_4B RES_CHECK=0 DIST_PLAN="patrickstar" bash run_bert.sh
```

即关闭精度校准，并运行大小为 4B 的 GPT2 模型。可选的 `MODEL_NAME` 详见 `pretrain_bert_demo.py`。
