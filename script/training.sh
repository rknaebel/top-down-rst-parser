#!/bin/bash
set -x

for x2y in d2e d2p d2s p2s s2e; do
    for i in {1..5}; do
        python3 -m rstparser.main train \
               --hidden 512 \
               --epochs 30 \
               --dropout 0.2 \
               --bert-model bert-base-cased \
               --batch-size 8 \
               --embed-batch-size 32 \
               --gate-embed \
               --parent-label-embed \
               --maximize-metric \
               --train-file train.$x2y.jsonl \
               --valid-file valid.$x2y.jsonl \
               --test-file test.jsonl \
               --serialization-dir models/$x2y.t$i \
               --hierarchical-type $x2y
    done
done
