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

