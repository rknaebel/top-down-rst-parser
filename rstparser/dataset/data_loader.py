import json
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import torch.utils.data
from nltk import Tree

from rstparser.dataset.tree_split import tree_division
from rstparser.dataset.trees import load_tree_from_string


class Sample:
    def __init__(self, doc_id, labelled_attachment_tree, raw_tokenized_strings, spans, starts_sentence,
                 starts_paragraph, parent_label):
        self.doc_id = doc_id
        self.tree = load_tree_from_string(labelled_attachment_tree)
        self.word = (None, len(raw_tokenized_strings), np.array([len(e) for e in raw_tokenized_strings]))
        self.edu_len = len(raw_tokenized_strings)
        self.words_len = np.array([len(e) for e in raw_tokenized_strings])
        self.elmo_word = raw_tokenized_strings
        self.spans = spans
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.parent_label = parent_label


class Batch:
    def __init__(self, doc_id, tree, word, edu_len, words_len, elmo_word, spans, starts_sentence, starts_paragraph,
                 parent_label):
        self.doc_id = doc_id
        self.tree = tree
        self.word = word
        self.edu_len = edu_len
        self.words_len = words_len
        self.elmo_word = elmo_word
        # TODO check if necessary, otherwise remove
        self.spans = spans
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.parent_label = parent_label

    @staticmethod
    def from_samples(samples: List[Sample]):
        edu_lengths = [s.word[1] for s in samples]
        word_lengths = np.zeros((len(samples), max(edu_lengths)))
        for i, s in enumerate(samples):
            word_lengths[i][:s.word[1]] = s.word[2]
        return Batch(
            doc_id=[s.doc_id for s in samples],
            tree=[s.tree for s in samples],
            word=(
                None,
                torch.tensor(edu_lengths),
                torch.from_numpy(word_lengths),
            ),
            edu_len=torch.tensor(edu_lengths),
            words_len=torch.from_numpy(word_lengths),
            elmo_word=[s.elmo_word for s in samples],
            spans=[s.spans for s in samples],
            starts_sentence=[s.starts_sentence for s in samples],
            starts_paragraph=[s.starts_paragraph for s in samples],
            parent_label=[s.parent_label for s in samples],
        )

    def __len__(self):
        return len(self.doc_id)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_files, config):
        self.ns_counter = Counter()
        self.relation_counter = Counter()

        self.items = []
        dataset = [json.loads(line) for data_file in data_files for line in open(data_file)]
        for item in dataset:
            self.count_relation_properties(item['labelled_attachment_tree'])

        if config.hierarchical_type == 'd2e':
            for item in dataset:
                self.items.append(Sample(item['doc_id'], item['labelled_attachment_tree'],
                                         item['raw_tokenized_strings'], item['spans'], item['starts_sentence'],
                                         item['starts_paragraph'], item['parent_label']))
        else:
            for item in tree_division(dataset, config.hierarchical_type):
                self.items.append(Sample(item['doc_id'], item['labelled_attachment_tree'],
                                         item['raw_tokenized_strings'], item['spans'], item['starts_sentence'],
                                         item['starts_paragraph'], item['parent_label']))

    def count_relation_properties(self, labelled_attachment_tree):
        attach_tree = Tree.fromstring(labelled_attachment_tree)
        labels = [attach_tree[p].label() for p in attach_tree.treepositions()
                  if not isinstance(attach_tree[p], str) and attach_tree[p].height() > 2]
        for label in labels:
            ns, relation = label.split(':')
            self.ns_counter[ns] += 1
            self.relation_counter[relation] += 1

    def get_vocabs(self, specials=None):
        specials = specials or []
        return Vocab(self.ns_counter, specials=specials), Vocab(self.relation_counter, specials=specials)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# TODO simplify VOCAB
class Vocab:
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    UNK = '<unk>'

    def __init__(self, counter, specials=('<unk>', '<pad>'),
                 specials_first=True):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ['<unk'>, '<pad>']
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
        """
        self.freqs = counter
        counter = counter.copy()

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        if Vocab.UNK in specials:  # hard-coded for now
            unk_index = specials.index(Vocab.UNK)  # position in list
            # account for ordering of specials, set variable
            self.unk_index = unk_index if specials_first else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        if state.get("unk_index", None) is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
