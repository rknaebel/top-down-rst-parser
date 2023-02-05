import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from rstparser.dataset.merge_file import Doc
from rstparser.networks.hierarchical import HierarchicalParser


def main():
    parser = argparse.ArgumentParser(description="span based rst parser")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--hierarchical-type', choices=['d2e', 'd2s2e', 'd2p2s2e'], required=True)
    parser.add_argument('--use-hard-boundary', action='store_true')
    parser.add_argument('--model-paths', required=True, nargs='+')
    parser.add_argument('--input-doc', required=True, nargs='+', type=Path)
    parser.add_argument('--output-dir', default='output', type=Path)
    config = parser.parse_args()

    model = HierarchicalParser.load_model(config.model_paths, config)

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
        tree_path = Path(config.output_dir / doc_path.name).with_suffix('.tree')
        if tree_path.exists() and tree_path.stat().st_size > 100:
            continue
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
            with open(tree_path, 'w') as f:
                f.write(tree.pformat(margin=256))


if __name__ == '__main__':
    main()
