import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data

from rstparser.dataset.preprocess import read_conll_file, preprocess


class Sample:
    def __init__(self, doc_id, tokens, starts_edu):
        self.doc_id = doc_id
        self.tokens = tokens
        self.starts_edu = np.array(starts_edu, dtype=bool)

    def __str__(self):
        return f"{self.doc_id}: " + " ".join(f"{t}-{int(s)}" for t, s in zip(self.tokens, self.starts_edu))


class Batch:
    def __init__(self, doc_id, sents_len, tokens, starts_edu):
        self.doc_id = doc_id
        self.sents_len = sents_len
        self.tokens = tokens
        self.starts_edu = starts_edu

    @staticmethod
    def from_samples(samples: List[Sample]):
        return Batch(
            doc_id=[s.doc_id for s in samples],
            sents_len=torch.tensor([len(s.tokens) for s in samples], dtype=torch.int16),
            tokens=[(s.tokens,) for s in samples],
            starts_edu=[torch.from_numpy(s.starts_edu).to(torch.float32) for s in samples]
        )

    def __str__(self):
        return str([self.doc_id, self.tokens, self.starts_edu])

    def __len__(self):
        return len(self.doc_id)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_files, conll_paths=()):
        self.items = []
        dataset = [json.loads(line) for data_file in data_files for line in open(data_file)]
        for conll_path in conll_paths:
            conll_path = Path(conll_path)
            for conll_i, conll_file in enumerate(conll_path.glob('*.conll')):
                doc = read_conll_file(conll_file.open())
                doc['doc_id'] = conll_path.name
                try:
                    doc = preprocess(doc)
                except:
                    print(doc, file=sys.stderr)
                    exit(1)
                dataset.append(doc)

        for item in dataset:
            tokens = []
            starts_edu = []
            for edu, start_sentence in zip(item['raw_tokenized_strings'], item['starts_sentence']):
                if start_sentence and tokens:
                    self.items.append(Sample(item['doc_id'], tokens, np.array(starts_edu, dtype=bool)))
                    tokens = []
                    starts_edu = []
                tokens.extend(edu)
                starts_edu.extend([True] + [False] * (len(edu) - 1))
                starts_edu[0] = False
            if tokens:
                self.items.append(Sample(item['doc_id'], tokens, np.array(starts_edu, dtype=bool)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
