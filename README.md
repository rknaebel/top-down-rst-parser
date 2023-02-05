# Neural-based Discourse Parser

This repository is based
on ["Top-down RST Parsing Utilizing Granularity Levels in Documents"](https://ojs.aaai.org/index.php/AAAI/article/view/6321)
.

## Training

### Data Preprocessing

Before running a script, you should organize files into a common SOURCE directory. This directory contains documents
in [CONLL-U format](https://universaldependencies.org/format.html) where the last column MISC contains information about
discourse segmentation. The field 'BeginSeg=YES' is set, if a new elementary discourse unit (EDU) begins. This
annotations must strictly correlate with the number of EDUs in the annotation file. For RST annotations, there are two
options:

1. _file.dis_: files corresponding to the original RST-DT tree format
2. _file.tree_: files corresponding to the labeled attachment tree format, used in this system

```bash
python -m rstparser.cli.preprocess SOURCE DESTINATION
```

During preprocessing, each document file is converted into corresponding jsonl format described below:

```bash
"doc_id": "wsj_****"
"labelled_attachment_tree": "(nucleus-satellite:Elaboration (text 0) (text 1))"
"tokenized_strings": ["first sentence corresponding to text 1 .", "and this is second sentence ."]
"raw_tokenized_strings": ["first", "sentence", "corresponding", "to", "text", "1", ".", "and", "this", "is", "second", "sentence", "."]
"starts_sentence": [true, true]
"starts_paragraph": [true, false]
"parent_label": null
"granularity_type": D2E
```

There are sample files of our preprocessing in `data/sample/`.

### Segmentation

Train a segmentation model:

```bash
python -m rstparser.networks.segmenter --train-file rst_data/train.jsonl --valid-file rst_data/valid.jsonl --batch-size 64 --hidden 128 --bert-model roberta-base --epochs 20 --test-file rst_data/test.jsonl --serialization-dir models/seg.t3
```

Test segmentation model:

```bash
python -m rstparser.networks.segmenter --batch-size 64 --test-file rst_data/test.jsonl --model-paths models/seg.*/model_best_*
```

### Top-Down

Train the model 5 times for D2E, D2P, D2S, P2S, P2E and S2E. If you need to select a GPU device, please use an
environment variable `CUDA_VISIBLE_DEVICES`.

```bash
bash script/training.sh
```

Evaluate on test set for D2E, D2S2E and D2P2S2E with 5 ensemble setting.

```bash
bash script/evaluate.sh
```

## Further References

- [Top-Down RST Parsing Utilizing Granularity Levels in Documents](https://ojs.aaai.org/index.php/AAAI/article/view/6321)
- [Toward Fast and Accurate Neural Discourse Segmentation](https://aclanthology.org/D18-1116/)
- [Improving Neural RST Parsing Model with Silver Agreement Subtrees](https://aclanthology.org/2021.naacl-main.127/)
