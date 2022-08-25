import torch
import torch.optim as optim
from nltk import Tree
from torch.utils.data import DataLoader

from rstparser.config import load_config
from rstparser.dataset.data_loader import Dataset, Batch
from rstparser.networks.hierarchical import HierarchicalParser
from rstparser.networks.parser import SpanBasedParser
from rstparser.trainer.trainer import Trainer


def main():
    config = load_config()
    if config.subcommand == 'train':
        train(config)
    elif config.subcommand == 'test':
        test(config)
    else:
        print('train / test')
        return -1

    return 0


def train(config):
    dataset = Dataset([config.train_file, config.valid_file], config)
    ns_vocab, rel_vocab = dataset.get_vocabs(['<pad>'])
    model = SpanBasedParser.build_model(config, ns_vocab, rel_vocab)
    optimizer = {'adam': optim.Adam, 'sgd': optim.SGD}[config.optimizer](
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    trainer = Trainer(config, model, optimizer, scheduler, dataset)
    trainer.run()


def test(config):
    dataset = Dataset([config.test_file], config)
    test_iter = DataLoader(dataset, batch_size=config.batch_size, collate_fn=Batch.from_samples)
    # model = SpanBasedParser.load_model(config.model_path[0], config)
    # model = EnsembleParser.load_model(config.model_path, config)
    model = HierarchicalParser.load_model(config.model_path, config)
    scores = Trainer.valid(model, test_iter)
    print(scores)

    doc_ids = []
    pred_trees = []
    for batch in test_iter:
        with torch.no_grad():
            output_dict = model(batch)

        doc_ids.extend(batch.doc_id)
        pred_trees.extend(output_dict['tree'])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pred_trees = [Tree.fromstring(tree) for tree in pred_trees]
    for doc_id, tree in zip(doc_ids, pred_trees):
        tree_path = config.output_dir / '{}.tree'.format(doc_id)
        with open(tree_path, 'w') as f:
            print(tree, file=f)


if __name__ == '__main__':
    main()
