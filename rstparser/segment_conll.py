import argparse
import itertools
import logging
from collections import OrderedDict
from pathlib import Path

import conllu
import numpy as np
import torch
from tqdm import tqdm

from rstparser.dataset.segment_loader import Sample, Batch
from rstparser.networks.segmenter import SegmenterEnsemble


def main():
    parser = argparse.ArgumentParser(description="discourse segmenter")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model-paths', nargs='*', default=[])

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--embed-batch-size', type=int, default=32)

    parser.add_argument('--keep-all-serialized-models', action='store_true')
    parser.add_argument('--serialization-dir', default='models/', type=Path)
    parser.add_argument('--log-file', default='training.log')
    parser.add_argument('--model-name', default='model')

    parser.add_argument('--input-doc', required=True, nargs='+', type=Path)
    parser.add_argument('--output-dir', default='output', type=Path)

    config = parser.parse_args()

    model = SegmenterEnsemble.load_model(config.model_paths, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if len(config.input_doc) == 1:
        input_doc = config.input_doc[0]
        if input_doc.is_dir():
            filelist = input_doc.glob('*.conll')
        else:
            filelist = [input_doc]
    else:
        filelist = config.input_doc

    for doc_path in tqdm(filelist):
        logging.debug(f'processing: {doc_path}')
        output_path = (config.output_dir / doc_path.name).with_suffix('.conll')
        with torch.no_grad():
            with open(output_path, 'w') as f:
                iter_sents = conllu.parse_incr(doc_path.open(), fields=conllu.parser.DEFAULT_FIELDS)
                iter_sents = filter(lambda d: len(d) < 300, iter_sents)
                while True:
                    sents_batch = list(itertools.islice(iter_sents, config.batch_size))
                    if not len(sents_batch):
                        break
                    batch = Batch.from_samples([Sample(doc_path.name,
                                                       [tok['form'] for tok in sent], np.array([]))
                                                for sent in sents_batch])
                    outputs = model.parse(batch)['edu_splits']
                    for tokens, preds, sent in zip(batch.tokens, outputs, sents_batch):
                        # pred_out = ' '.join((" || " if p else "") + t for t, p in zip(tokens[0], preds))
                        # print('=' * 10)
                        # print('Pred EDU seg: {}'.format(pred_out))
                        for t_i, (tok, edu_start) in enumerate(zip(sent, preds)):
                            if t_i == 0 or edu_start.item() > 0.5:
                                if 'misc' in tok:
                                    tok['misc']['BeginSeg'] = 'YES'
                                else:
                                    tok['misc'] = OrderedDict({
                                        'BeginSeg': 'YES'
                                    })

                        f.write(sent.serialize())


if __name__ == '__main__':
    main()
