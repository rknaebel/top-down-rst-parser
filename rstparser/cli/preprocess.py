import json
import random
import re
from pathlib import Path

import click
import conllu
from tqdm import tqdm

import rstparser.dataset.tree_utils
from rstparser.dataset.tree_split import tree_division, make_text_span


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

    # 文の開始でないのに段落の開始である場合はFalseにする
    edu_starts_paragraph = [start_s and start_p for start_s, start_p in zip(edu_starts_sentence, edu_starts_paragraph)]

    # hdfファイルに書き出されたベクトルへのindexとして使う
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
    pidx = 0
    edu_i = 0
    edu_starts_paragraph = []
    tokens = []
    for sent_i, sent in enumerate(conllu.parse_incr(doc_file, fields=conllu.parser.DEFAULT_FIELDS)):
        if sent.metadata.get('newpar id'):
            pidx += 1
        for tok_i, tok in enumerate(sent):
            if tok.get('misc') and tok['misc'].get('BeginSeg') == 'YES':
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


def load_rst_data(source, tree_format='dis'):
    dis_files = Path(source).glob(f"*.{tree_format}")
    for dis_file in dis_files:
        dis_id = dis_file.stem.split('.', maxsplit=1)[0]
        if dis_file.exists():
            with dis_file.open() as fh:
                if tree_format == "dis":
                    edus, tree = rstparser.dataset.tree_utils.preprocess_rst_dis_format(
                        re.sub(r"\s+", " ", fh.read()).strip())
                elif tree_format == 'tree':
                    tree = fh.read().replace('\n', '').strip()
        else:
            tree = ""
        with dis_file.with_suffix('.conll').open() as fh:
            doc = read_conll_file(fh)
        doc['rst_tree'] = tree
        doc['doc_id'] = dis_id
        yield doc


@click.command()
@click.argument("source", type=click.Path(file_okay=False, exists=True))
@click.argument("target", type=click.Path(file_okay=False))
@click.option("--target-split", type=click.Path(file_okay=False))
@click.option("--split-ratio", default=0.8, type=float)
@click.option("--random-seed", default=1234567890, type=int)
@click.option("--divide", is_flag=True)
def main(source, target, target_split, split_ratio, random_seed, divide):
    dataset = list(tqdm(map(preprocess, load_rst_data(source))))

    if target_split:
        random.seed(random_seed)
        random.shuffle(dataset)
        split_point = int(len(dataset) * split_ratio)
        datasets = [(target, dataset[:split_point]), (target_split, dataset[split_point:])]
    else:
        datasets = [(target, dataset)]

    for path, ds in datasets:
        Path(path).parent.mkdir(exist_ok=True)
        suffix = '.d2e.jsonl' if divide else '.jsonl'
        with Path(path).with_suffix(suffix).open('w') as fh:
            for d in ds:
                fh.write(json.dumps(d) + '\n')

        if divide:
            for x2y in ['d2p', 'd2s', 'p2e', 'p2s', 's2e']:
                x2y_dataset = tree_division(ds, x2y)
                with Path(path).with_suffix(f'.{x2y}.jsonl').open('w') as fh:
                    for d in x2y_dataset:
                        fh.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    main()
