import json
import random
import re
from pathlib import Path
from pprint import pprint

import click
from tqdm import tqdm

import rstparser.dataset.tree_utils
from rstparser.dataset.preprocess import read_conll_file, preprocess
from rstparser.dataset.tree_split import tree_division


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
        if not len(edus) == len(doc['edu_start_indices']):
            print(f"{dis_id}: {len(edus)} {len(doc['edu_start_indices'])}")
            print("LENGTH SENTS", len(doc['tokens']))
            print("DIS EDUS:")
            pprint(edus, width=120)
            print("DOC INDICES:")
            pprint(doc['edu_start_indices'], width=120)
            exit(1)
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
