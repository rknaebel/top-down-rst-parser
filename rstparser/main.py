import torch.optim as optim
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


def train(config):
    dataset = Dataset(config.train_file, config)
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
    model = HierarchicalParser.load_model(config.model_path, config)
    scores = Trainer.valid(model, test_iter)
    print("Evaluation")
    print("> Number of instances:", len(dataset.items))
    print("> Dataset Relations", dataset.relation_counter)
    print("    Span:", scores['span'])
    print("    Nuclearity:", scores['ns'])
    print("    Relation:", scores['relation'])
    print("    Full:", scores['full'])


if __name__ == '__main__':
    main()
