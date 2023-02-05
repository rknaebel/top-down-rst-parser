import conllu

from rstparser.dataset.data_loader import Sample, Batch


class Doc:
    def __init__(self, tokens, starts_sentence, starts_edu, starts_paragraph, parent_label, doc_id=None):
        self.tokens = tokens
        self.starts_edu = starts_edu
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.parent_label = parent_label
        self.doc_id = doc_id

    def __repr__(self):
        n_tokens = 10
        tokens = 'tokens   :\t{} ... {}'.format(self.tokens[:n_tokens], self.tokens[-n_tokens:])
        edu_flags = 'edu flag :\t{} ... {}'.format(self.starts_edu[:n_tokens], self.starts_edu[-n_tokens:])
        sent_flags = 'sent flag:\t{} ... {}'.format(self.starts_sentence[:n_tokens], self.starts_sentence[-n_tokens:])
        para_flags = 'para flag:\t{} ... {}'.format(self.starts_paragraph[:n_tokens], self.starts_paragraph[-n_tokens:])
        label = 'parent_label:\t{}'.format(self.parent_label)
        txt = '\n'.join([tokens, edu_flags, sent_flags, para_flags, label])
        return txt

    def to_batch(self, parent_label=None, x2y='d2e', index=0, no_batch=False):
        start_offset = 0

        if x2y in ['d2e', 'd2p', 'd2s']:
            starts_sentence = self.starts_sentence
            starts_paragraph = self.starts_paragraph
            tokens = self.tokens
            if x2y == 'd2e':
                starts_xxx = self.starts_edu
            elif x2y == 'd2s':
                starts_xxx = self.starts_sentence
            else:
                starts_xxx = self.starts_paragraph
        elif x2y in ['p2s', 'p2e', 's2e']:
            def get_span(hierarchical_type, word_starts_sentence, word_starts_paragraph, index):
                if hierarchical_type.startswith('p'):
                    flags = word_starts_paragraph
                else:
                    flags = word_starts_sentence

                # flags: [True, False, False, False, True, False, True, ...]
                # index: 0 -> start, end = 0, 4
                # index: 1 -> start, end = 4, 6
                count = 0
                start, end = 0, len(flags)
                for i, f in enumerate(flags):
                    if not f:
                        continue
                    if count == index:
                        start = i
                    if count == index + 1:
                        end = i
                        break
                    count += 1
                return start, end

            start, end = get_span(x2y, self.starts_sentence, self.starts_paragraph, index)
            start_offset = start  # to make text span
            starts_sentence = self.starts_sentence[start:end]
            starts_paragraph = self.starts_paragraph[start:end]
            tokens = self.tokens[start:end]

            if x2y == 'p2s':
                starts_xxx = self.starts_sentence[start:end]
            else:
                starts_xxx = self.starts_edu[start:end]
        else:
            raise ValueError('Unknown hierarchical type')
        # tokenized_strings: List of edus
        # raw_tokenized_strings: List of edus splitted by white-space
        # starts_*: List of bool value representing start of *
        tokenized_strings = self.make_edu(tokens, starts_xxx)
        raw_tokenized_strings = [edu.split() for edu in tokenized_strings]
        starts_sentence = self.make_starts(starts_xxx, starts_sentence)
        starts_paragraph = self.make_starts(starts_xxx, starts_paragraph)
        parent_label = self.parent_label if parent_label is None else parent_label
        spans, _ = self.make_text_span(tokenized_strings, start_offset, self.doc_id)

        assert len(tokenized_strings) == len(starts_sentence) == len(starts_paragraph), 'num input seqs not same'

        example = Sample(
            doc_id=self.doc_id,
            labelled_attachment_tree='(nucleus-nucleus:Elaboration (text 1) (text 2))',  # DummyTree
            raw_tokenized_strings=raw_tokenized_strings,
            spans=spans,
            starts_sentence=starts_sentence,
            starts_paragraph=starts_paragraph,
            parent_label=parent_label,
        )
        if no_batch:
            return example
        else:
            return Batch.from_samples([example])

    def make_edu(self, tokens, is_starts):
        assert len(tokens) == len(is_starts), f"{len(tokens)} -- {len(is_starts)}"
        edu_strings = []
        edu = []
        for token, is_start in zip(tokens + [''], is_starts + [True]):
            if is_start and edu:
                edu_strings.append(' '.join(edu))
                edu = []
            edu.append(token)

        return edu_strings

    def make_starts(self, base, target):
        starts = []
        for a, b in zip(base, target):
            if a:
                starts.append(b)

        return starts

    def make_text_span(self, edu_strings, starts_offset=0, doc_id=None):
        spans = []
        offset = starts_offset
        for edu_stirng in edu_strings:
            words = edu_stirng.split()
            n_words = len(words)
            if doc_id is not None:
                span = (offset, offset + n_words, doc_id)
            else:
                span = (offset, offset + n_words)
            spans.append(span)
            offset += n_words

        assert len(spans) == len(edu_strings)
        return spans, offset

    @staticmethod
    def from_batch(batch: Batch):
        assert len(batch) == 1
        tokenized_edu_strings = batch.elmo_word[0]
        edu_starts_sentence = batch.starts_sentence[0]  # edu_starts_sentence
        edu_starts_paragraph = batch.starts_paragraph[0]  # edu_starts_paragraph
        assert len(tokenized_edu_strings) == len(edu_starts_sentence) == len(edu_starts_paragraph)
        doc_id = batch.doc_id[0]
        parent_label = batch.parent_label[0]
        # edu -> word
        tokens = sum(tokenized_edu_strings, [])
        starts_edu = sum([[True] + [False] * (len(edu) - 1) for edu in tokenized_edu_strings], [])
        starts_sentence = sum([[is_start] + [False] * (len(edu) - 1) for edu, is_start in
                               zip(tokenized_edu_strings, edu_starts_sentence)], [])
        starts_paragraph = sum([[is_start] + [False] * (len(edu) - 1) for edu, is_start in
                                zip(tokenized_edu_strings, edu_starts_paragraph)], [])
        assert len(tokens) == len(starts_edu) == len(starts_sentence) == len(starts_paragraph)
        return Doc(tokens, starts_sentence, starts_edu, starts_paragraph, parent_label, doc_id)

    @staticmethod
    def from_conll_file(doc_file):
        edu_start_indices = []
        edu_i = 0
        edu_starts_paragraph = []
        tokens = []
        for sent_i, sent in enumerate(conllu.parse_incr(doc_file, fields=conllu.parser.DEFAULT_FIELDS)):
            for tok_i, tok in enumerate(sent):
                if tok_i == 0 or tok.get('misc') and tok['misc'].get('BeginSeg') == 'YES':
                    edu_i += 1
                    edu_starts_paragraph.append('newpar id' in sent.metadata)
                    edu_start_indices.append((sent_i, tok_i, edu_i))
            tokens.append([tok['form'] for tok in sent])

        tokenized_edu_strings = []
        edu_starts_sentence = []
        sentence_id, token_id, edu_id = edu_start_indices[0]
        for next_sentence_id, next_token_id, next_edu_id in edu_start_indices[1:] + [(-1, -1, -1)]:
            end_token_id = next_token_id if token_id < next_token_id else None
            tokenized_edu_strings.append(tokens[sentence_id][token_id: end_token_id])
            edu_starts_sentence.append(token_id == 0)

            sentence_id = next_sentence_id
            token_id = next_token_id

        assert len(tokenized_edu_strings) == len(edu_starts_sentence) == len(edu_starts_paragraph)

        edu_starts_paragraph = [start_s and start_p for start_s, start_p in
                                zip(edu_starts_sentence, edu_starts_paragraph)]

        starts_edu = sum([[True] + [False] * (len(edu) - 1) for edu in tokenized_edu_strings], [])
        starts_sentence = sum([[is_start] + [False] * (len(edu) - 1) for edu, is_start in
                               zip(tokenized_edu_strings, edu_starts_sentence)], [])
        starts_paragraph = sum([[is_start] + [False] * (len(edu) - 1) for edu, is_start in
                                zip(tokenized_edu_strings, edu_starts_paragraph)], [])
        tokens = sum(tokens, [])

        return Doc(tokens, starts_sentence, starts_edu, starts_paragraph, parent_label=None)
