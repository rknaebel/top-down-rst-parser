#!/bash/bin
set -x

# RSTDT=path/to/RSTDT
RSTDT=~/resource/Heilman_tb

DATA=data
mkdir -p $DATA

python -m rstparser.cli.preprocess \
       -src $RSTDT/rst_discourse_tb_edus_TEST.json \
       -tgt $DATA/test

python -m rstparser.cli.preprocess \
       -src $RSTDT/rst_discourse_tb_edus_TRAINING_DEV.json \
       -tgt $DATA/valid \
       -divide

python -m rstparser.cli.preprocess \
       -src $RSTDT/rst_discourse_tb_edus_TRAINING_TRAIN.json \
       -tgt $DATA/train \
       -divide

python -m rstparser.cli.label_vocab \
       --train $DATA/train.d2e.jsonl \
       --ns-vocab $DATA/ns.vocab \
       --relation-vocab $DATA/relation.vocab

python -m rstparser.cli.word_vocab \
       --train $DATA/train.d2e.jsonl \
       --valid $DATA/valid.d2e.jsonl \
       --test $DATA/test.jsonl \
       --vocab $DATA/word_vocab_full.pickle

python -m rstparser.cli.word_vocab \
       --test $DATA/test.jsonl \
       --vocab $DATA/word_vocab_test.pickle

python -m rstparser.cli.make_hdf \
       --json-file $DATA/train.d2e.jsonl $DATA/valid.d2e.jsonl $DATA/test.jsonl \
       --hdf-file $DATA/vectors.hdf \
       --vocab-file $DATA/word_vocab_full.pickle
       
