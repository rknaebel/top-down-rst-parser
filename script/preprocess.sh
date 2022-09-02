#!/bash/bin
set -x

RSTDT=~/resource/rst2

DATA=data
mkdir -p $DATA

python -m rstparser.cli.preprocess $RSTDT/test $DATA/test

python -m rstparser.cli.preprocess \
       $RSTDT/train \
       $DATA/valid \
       --target-split