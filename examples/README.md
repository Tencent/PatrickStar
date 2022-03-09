## PatrickStar examples

### Use PatrickStar with HuggingFace

`huggingface_bert.py` is a fine-tuning Huggingface example with Patrickstar. Could you compare it with the [official Huggingface example](https://huggingface.co/transformers/custom_datasets.html#seq-imdb) to know how to apply PatrickStar to existed projects.

Before running the example, you need to prepare the data:

```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

And change the directory used in `get_dataset()`. After these, you are ready to go:

```bash
python huggingface_bert.py
```

### Use PatrickStar to train large model

`run_transformers.sh` and `pretrain_demo.py` is an example to train large PTMs with PatrickStar. You could run different size of model by adding config to`run_transformers.sh`.

The following command will run a model with 4B params:

```bash
MODEL_NAME=GPT2_4B bash run_transformers.sh
```

For the available `MODEL_NAME`, please check `pretrain_demo.py`.

Check the accuracy of PatrickStar with Bert:

```bash
RES_CHECK=1 bash run_transformers.sh
```
