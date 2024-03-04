import argparse
import csv
import sys
from pathlib import Path

import conllu
import nltk
import torch
from tqdm import tqdm

from rstparser.dataset.merge_file import Doc
from rstparser.networks.hierarchical import HierarchicalParser


def write_csv(filelist, model):
    writer = csv.writer(sys.stdout)
    writer.writerow(('doc_id', 'prob', 'tree'))
    for doc_path in tqdm(filelist):
        with torch.no_grad():
            try:
                doc = Doc.from_conll_file(doc_path.open())
            except ValueError as e:
                sys.stderr.write(f"Error: {e} (skip)")
                continue
            try:
                tree = model.parse(doc)
            except RuntimeError as e:
                sys.stderr.write(f"Runtime Error: {e} at document {doc_path.name}")
                exit(1)
            writer.writerow((doc_path.stem, 0.0, tree))  # ._pformat_flat("", "()", False)
        sys.stdout.flush()


def write_files(filelist, model, output_dir):
    for doc_path in tqdm(filelist):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tree_path = (output_dir / doc_path.name).with_suffix('.tree')
        if tree_path.exists() and tree_path.stat().st_size > 100:
            continue
        with torch.no_grad():
            try:
                doc = Doc.from_conll_file(doc_path.open())
            except ValueError as e:
                sys.stderr.write(f"Error: {e} (skip)")
                continue
            try:
                tree, probs = model.parse(doc)
                tree = nltk.Tree.fromstring(tree)
                doc = conllu.parse(doc_path.open().read(), fields=conllu.parser.DEFAULT_FIELDS)
                edus = []
                edu_cur = []
                for sent in doc:
                    for tok in sent:
                        if tok['misc'].get('BeginSeg', 'NO') == 'YES' and len(edu_cur):
                            edus.append(f"edu_id={len(edus)},start={edu_cur[0]['misc']['StartChar']},"
                                        f"end={edu_cur[-1]['misc']['EndChar']}")
                            edu_cur = []
                        edu_cur.append(tok)
                edus.append(f"edu_id={len(edus)},start={edu_cur[0]['misc']['StartChar']},"
                            f"end={edu_cur[-1]['misc']['EndChar']}")

                for subtree in tree.subtrees():
                    if isinstance(subtree[0], str):
                        edu_index = int(subtree[0])
                        subtree.clear()
                        subtree.append(edus[edu_index])
                tree = tree._pformat_flat("", "()", False)

            except RuntimeError as e:
                sys.stderr.write(f"Runtime Error: {e} at document {doc_path.name}")
                exit(1)
            with open(tree_path, 'w') as f:
                f.write(tree)


def main():
    parser = argparse.ArgumentParser(description="span based rst parser")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--hierarchical-type', choices=['d2e', 'd2s2e', 'd2p2s2e'], required=True)
    parser.add_argument('--use-hard-boundary', action='store_true')
    parser.add_argument('--model-paths', required=True, nargs='+')
    parser.add_argument('--input-doc', required=True, nargs='+', type=Path)
    parser.add_argument('--output-dir', default='-', type=str)
    config = parser.parse_args()

    model = HierarchicalParser.load_model(config.model_paths, config)

    if len(config.input_doc) == 1:
        input_doc = config.input_doc[0]
        if input_doc.is_dir():
            filelist = input_doc.glob('*.conll')
        else:
            filelist = [input_doc]
    else:
        filelist = config.input_doc

    if config.output_dir == '-':
        write_csv(filelist, model)
    else:
        write_files(filelist, model, config.output_dir)


if __name__ == '__main__':
    main()
