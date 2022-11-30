![XFACT Logo](https://xfact.net/static/img/logo.png)
# XNLP - Modular NLP Library

## Why?
Our current research bottleneck is adapting code and frameworks for each project. 
Even though we use the same datasets, there are several types of required preprocessing and 
experiment configurations that should be tested. 

This library wraps around the HuggingFace transformers library to help make the code modular and
make the experiments easy to configure, manage, and reproduce.

## Key concepts

### Everything is modular
File Readers, Datasets, Models, Training Regimens can all be separated into individual modules and 
run in any configuration together. 

### All modules are registrable
All the modules are registered in the library. Pipelines of modules can be configured by combining
the registered modules.

Vision:
```
deardr:
    reader: deardr_pt
    validation_reader: fever
    model: t5-base
    trainer: prefixdecoding
```


## Usage
Train file can be a file on system `/data/train.jsonl` 
or a huggingface dataset with a split from huggingface `:train` `:test`, or `validation`. 
E.g.: `nlpaueb/finer-139:test`

```
python -m xfact.train \ 
  --model_name_or_path t5-base \
  --reader deardr_pthl \
  --dataset deardr \
  --scorer SCORER \
  --postprocessing default \
  --output_dir SAVE_LOCATION \
  --train_file TRAIN_FILE \ 
  --validation_file VALIDATION_FILE \ 
  --do_train \
  --do_validation \
  --overwrite_output_dir \
  --eval_steps 500 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --learning_rate 5e-5 \
  --scheduler constant_with_warmup
```

The processing pipeline is:

* **Reader**: Gets the raw data form JSON and generates instances
* **Dataset**: Takes an instance and prepares it for input into a model
  * There are 3 possible choices of dataset that can be extended:
  * `XFactSeq2SeqDataset` - a dataset which inputs a sequence and outputs a sequence
  * `XFactTaggingDataset` - a dataset which inputs a sequence and outputs a label for each token
  * `XFactClassificationDataset` - a dataset which inputs a sequence and outputs a single label
* **Model**: Predicts Sequence, Tags, or Label
* **PostProcessing**: Takes output of model and makes the output ready for scorer
* **Scorer**: Takes the postprocessed outputs and compute metrics with it

#### Reader
The reader can be implemented in any class registered 
```
--reader ABCDEF
```

The Reader overrides a class that generates a sequence of instances
```
@Reader.register("ABCDEF")
class ABCReader(Reader):
  def generate_instances(self, instance):
    ...
    yield {
        "source": "hello",
        "target": "안녕하세요"
    }
    
```

#### Dataset
The dataset can be implemented in any class registered 
```
--dataset XYZDataset
```

The Dataset overrides a class that generates source and target for seq2seq
```
@XFactDataset.register("XYZDataset")
class XYZDataset(XFactSeq2SeqDataset):
  def prepare_src(self, instance):
    return "Translate from English to Korean: " + instance["source"]
    
  def prepare_tgt(self, instance):
    return instance["target"]
```