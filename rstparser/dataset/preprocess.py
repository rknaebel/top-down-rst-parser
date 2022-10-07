import logging

import conllu

from rstparser.dataset.tree_split import make_text_span


def preprocess(src):
    doc_id = src['doc_id']
    tokenized_edu_strings = []
    edu_starts_sentence = []
    edu_starts_paragraph = src['edu_starts_paragraph']
    tokens = src['tokens']
    edu_start_indices = src['edu_start_indices']
    sentence_id, token_id, edu_id = edu_start_indices[0]
    for next_sentence_id, next_token_id, next_edu_id in edu_start_indices[1:] + [(-1, -1, -1)]:
        end_token_id = next_token_id if token_id < next_token_id else None
        tokenized_edu_strings.append(' '.join(tokens[sentence_id][token_id: end_token_id]))
        edu_starts_sentence.append(token_id == 0)

        sentence_id = next_sentence_id
        token_id = next_token_id

    assert len(tokenized_edu_strings) == len(edu_starts_sentence) == len(edu_starts_paragraph)

    # False if it is the start of a paragraph but not the start of a sentence
    edu_starts_paragraph = [start_s and start_p for start_s, start_p in zip(edu_starts_sentence, edu_starts_paragraph)]

    text_span, _ = make_text_span(tokenized_edu_strings, doc_id=doc_id)

    return {
        'doc_id': doc_id,
        'labelled_attachment_tree': src['rst_tree'],
        'tokenized_strings': tokenized_edu_strings,
        'raw_tokenized_strings': [edu_string.split() for edu_string in tokenized_edu_strings],
        'spans': text_span,
        'starts_sentence': edu_starts_sentence,
        'starts_paragraph': edu_starts_paragraph,
        'parent_label': None,
        'granularity_type': 'D2E',
    }


def read_conll_file(doc_file):
    indices = []
    gidx = 0
    edu_i = 0
    edu_starts_paragraph = []
    tokens = []
    for sent_i, sent in enumerate(conllu.parse_incr(doc_file, fields=conllu.parser.DEFAULT_FIELDS)):
        if len(sent) > 300:
            logging.warning("Skip sentence: too long.")
            continue
        for tok_i, tok in enumerate(sent):
            if tok_i == 0 or tok.get('misc') and tok['misc'].get('BeginSeg') == 'YES':
                edu_i += 1
                edu_starts_paragraph.append('newpar id' in sent.metadata)
                indices.append((sent_i, tok_i, edu_i))
            gidx += 1
        tokens.append([tok['form'] for tok in sent])
    return {
        'tokens': tokens,
        'edu_starts_paragraph': edu_starts_paragraph,
        'edu_start_indices': indices,
        'rst_tree': None,
    }
