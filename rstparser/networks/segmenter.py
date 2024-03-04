import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rstparser.dataset.segment_loader import Batch, Dataset
from rstparser.networks.embedder import TextEmbedder
from rstparser.networks.layers import BiLSTM, FeedForward
from rstparser.trainer.checkpointer import Checkpointer
from rstparser.trainer.score import Score


class Segmenter(nn.Module):
    def __init__(self, text_embedder: TextEmbedder, hidden_size, dropout, device):
        super(Segmenter, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        # Embeddings
        self.text_embedder: TextEmbedder = text_embedder
        embed_size = self.text_embedder.word_embedder.get_embed_size()
        # BiLSTM
        self.bilstm_embd = BiLSTM(embed_size, self.hidden_size, self.dropout)
        self.bilstm_out = BiLSTM(self.hidden_size * 2, self.hidden_size, self.dropout)
        self.f_out = FeedForward(self.hidden_size * 2, [self.hidden_size], 2, self.dropout)
        # (Softmax + Entropy)-Loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def freeze(self):
        self.text_embedder.freeze()
        self.bilstm_embd.freeze()
        self.bilstm_out.freeze()

    @staticmethod
    def build_model(config):
        embedder = TextEmbedder.build_model(config)
        hidden = config.hidden
        dropout = config.dropout
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        model = Segmenter(embedder, hidden, dropout, device)
        model.to(device)
        return model

    @staticmethod
    def load_model(model_path, config):
        print('load model: {}'.format(model_path), file=sys.stderr)
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        model_state = Checkpointer.restore(model_path, device=device)
        model_config = model_state['config']
        model_config.cpu = config.cpu
        model_param = model_state['model']
        model = Segmenter.build_model(model_config)
        model.load_state_dict(model_param, strict=False)
        model.eval()
        return model

    # def parse(self, doc):
    #     batch = doc.to_batch(self.device)
    #     output = self.forward(batch)
    #     tree = output['tree'][0]
    #     return tree

    def forward(self, batch):
        rnn_outputs = self.embed(batch)
        output_logits = self.bilstm_out(rnn_outputs, batch.sents_len)
        output_logits = self.f_out(output_logits)
        losses = []
        outputs = []
        for pred, sent_len, gold in zip(output_logits, batch.sents_len, batch.starts_edu):
            gold = gold.to(torch.long).to(self.device)
            pred = pred[:sent_len]
            losses.append(self.loss_fn(pred, gold))
            outputs.append(pred.argmax(-1))

        loss = torch.mean(torch.stack(losses))

        return {
            'loss': loss,
            'edu_splits': outputs,
        }

    def parse(self, batch):
        rnn_outputs = self.embed(batch)
        output_logits = self.bilstm_out(rnn_outputs, batch.sents_len)
        output_logits = self.f_out(output_logits)
        outputs = []
        for pred, sent_len in zip(output_logits, batch.sents_len):
            pred = pred[:sent_len]
            outputs.append(pred.argmax(-1))

        return {
            'edu_splits': outputs,
        }

    def embed(self, batch: Batch):
        starts = [[True] for _ in range(len(batch))]
        edu_embeddings = self.text_embedder.embed_sequence(batch.tokens, starts)
        lstm_outputs = self.bilstm_embd(edu_embeddings, batch.sents_len)
        return lstm_outputs


class SegmenterEnsemble(nn.Module):

    def __init__(self, models):
        super(SegmenterEnsemble, self).__init__()
        self.models: List[Segmenter] = models

    def freeze(self):
        for model in self.models:
            model.freeze()

    @staticmethod
    def load_model(model_paths, config):
        if len(model_paths) == 0:
            raise ValueError('No models loaded...')
        models = []
        for model_path in model_paths:
            model = Segmenter.load_model(model_path, config)
            models.append(model)
        return SegmenterEnsemble(models)

    def forward(self, batch):
        outputs = []
        loss = []
        for model in self.models:
            out = model(batch)
            outputs.append(out['edu_splits'])
            loss.append(out['loss'].cpu())
        return {
            'loss': torch.stack(loss).mean(),
            'edu_splits': [torch.stack(preds).cpu().to(torch.float32).mean(0).round()
                           for preds in zip(*outputs)]
        }

    def parse(self, batch):
        outputs = []
        for model in self.models:
            out = model.parse(batch)
            outputs.append(out['edu_splits'])
        return {
            'edu_splits': [torch.stack(preds).cpu().to(torch.float32).mean(0).round()
                           for preds in zip(*outputs)]
        }


class Trainer:
    def __init__(self, config, model, train_dataset, valid_dataset=None):
        self._epochs = config.epochs
        self._disable_tqdm = False
        self._log_file = config.log_file
        self._serialization_dir = Path(config.serialization_dir)
        self._checkpointer = Checkpointer(self._serialization_dir,
                                          config.keep_all_serialized_models,
                                          config.model_name)

        self._score = Score('f1', save_minimum=False)
        if valid_dataset:
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
        else:
            dataset_length = len(train_dataset)
            train_size = int(dataset_length * config.data_split)
            valid_size = dataset_length - train_size
            self.train_dataset, self.valid_dataset = random_split(self.train_dataset, [train_size, valid_size])

        self._model = model
        self._optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self._max_grad_norm = config.grad_clipping
        self._scheduler = optim.lr_scheduler.ExponentialLR(self._optimizer, config.lr_decay)

        self._config = config
        self._cur_epoch = 1
        self.batch_size = config.batch_size

    def load_checkpoint(self, best_checkpoint=False):
        checkpoint_path = self._checkpointer.get_latest_checkpoint(best_checkpoint=best_checkpoint)
        device = torch.device('cpu') if self._config.cpu else torch.device('cuda:0')
        checkpoint = self._checkpointer.restore(checkpoint_path, device)
        self._cur_epoch = checkpoint['epoch'] + 1
        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._scheduler.load_state_dict(checkpoint['sched'])
        self._score.load_state_dict(checkpoint['score'])
        print('train from checkpoint: {}'.format(checkpoint_path), file=sys.stderr)

    def run(self, epochs=0):
        run_epochs = epochs if epochs > 0 else self._epochs
        train_iter = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=Batch.from_samples)
        valid_iter = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=Batch.from_samples)
        for epoch in range(self._cur_epoch, self._cur_epoch + run_epochs):
            train_loss = self.train(self._model, train_iter, self._optimizer,
                                    self._max_grad_norm, self._disable_tqdm)
            valid_score = self.valid(self._model, valid_iter,
                                     # print_every_n_batch=10, print_result=True,
                                     disable_tqdm=self._disable_tqdm)
            self._scheduler.step()

            self._score.append(valid_score, epoch)
            self._save(epoch, self._score.is_best(epoch))

            scores = {
                'train/loss': train_loss,
                'valid/score': valid_score,
            }
            self._report(epoch, scores)

        return

    @staticmethod
    def train(model, _iter, optimizer, max_grad_norm, disable_tqdm=False):
        total_loss = 0
        total_norm = 0
        model.train()
        for batch in tqdm(_iter, desc='training', ncols=128, disable=disable_tqdm):
            optimizer.zero_grad()
            output_dict = model(batch)
            loss = output_dict["loss"]
            if loss.item() == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            loss = output_dict["loss"]
            total_loss += loss.item() * len(batch)
            total_norm += len(batch)

        return {'loss': total_loss / total_norm}

    @staticmethod
    def valid(model, _iter, print_every_n_batch=0, print_result=False, disable_tqdm=False):
        model.eval()
        total_loss = 0
        total_norm = 0
        total_cnt, pred_cnt, correct_cnt = 0, 0, 0
        for batch_i, batch in tqdm(enumerate(_iter), total=len(_iter), desc='validation',
                                   ncols=128, disable=disable_tqdm):
            with torch.no_grad():
                output_dict = model(batch)
                for tokens, preds, splits in zip(batch.tokens, output_dict['edu_splits'], batch.starts_edu):
                    # convert predictions into split positions
                    gold_segs = np.where(splits.cpu().numpy() > 0)[0]
                    pred_segs = np.where(preds.cpu().numpy() > 0)[0]
                    # count for metrics
                    total_cnt += len(gold_segs)
                    pred_cnt += len(pred_segs)
                    correct_cnt += len(set(gold_segs) & set(pred_segs))
                    # print if wanted
                    if print_result and print_every_n_batch > 0 and batch_i % print_every_n_batch == 0:
                        if np.absolute(preds.cpu().numpy() - splits.numpy()).sum():
                            pred_out = ' '.join((" || " if p else "") + t for t, p in zip(tokens[0], preds))
                            gold_out = ' '.join((" || " if p else "") + t for t, p in zip(tokens[0], splits))
                            print('=' * 10)
                            print('Gold EDU seg: {}'.format(gold_out))
                            print('Pred EDU seg: {}'.format(pred_out))
                            print('=' * 10)
                loss = output_dict["loss"]
                total_loss += loss.item() * len(batch)
                total_norm += len(batch)
        perf = {
            'loss': total_loss / total_norm,
            'precision': 1.0 * correct_cnt / pred_cnt if pred_cnt > 0 else 0.0,
            'recall': 1.0 * correct_cnt / total_cnt if total_cnt > 0 else 0.0
        }
        if perf['precision'] > 0 and perf['recall'] > 0:
            perf['f1'] = 2.0 * perf['precision'] * perf['recall'] / (perf['precision'] + perf['recall'])
        else:
            perf['f1'] = 0
        return perf

    def _save(self, epoch, is_best):
        model_state = {
            "epoch": epoch,
            "model": self._model.state_dict(),
            "optim": self._optimizer.state_dict(),
            "sched": self._scheduler.state_dict(),
            "score": self._score.state_dict(),
            "config": vars(self._config),
        }
        self._checkpointer.save(epoch, model_state, is_best)
        return

    def get_best_model_path(self):
        return self._checkpointer.get_best_model_path()

    def _report(self, epoch, scores):
        json_text = json.dumps({'epoch': epoch, 'scores': scores}, indent=4)
        print(json_text, file=sys.stderr)

        with open(self._serialization_dir / self._log_file, "a") as f:
            print(json_text, file=f)

        return


def main():
    parser = argparse.ArgumentParser(description="discourse segmenter")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model-paths', nargs='*', default=[])
    parser.add_argument('--train-file', type=str, default="")
    parser.add_argument('--valid-file', type=str, default="")
    parser.add_argument('--conll-paths', nargs='*', default=[])
    parser.add_argument('--test-file', type=str, default="")

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--grad-clipping', type=float, default=5.0)
    parser.add_argument('--data-split', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--bert-model', type=str, default='bert-base-cased')
    parser.add_argument('--last-hidden-only', action='store_true')
    parser.add_argument('--embed-batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--keep-all-serialized-models', action='store_true')
    parser.add_argument('--serialization-dir', default='models/', type=Path)
    parser.add_argument('--log-file', default='training.log')
    parser.add_argument('--model-name', default='segment')

    config = parser.parse_args()

    if config.train_file and config.valid_file:
        dataset = Dataset([config.train_file], config.conll_paths)
        valid_dataset = Dataset([config.valid_file])
        model = Segmenter.build_model(config)

        trainer = Trainer(config, model, dataset, valid_dataset)
        trainer.run()
    else:
        print("missing train/valid data")

    if config.test_file:
        dataset = Dataset([config.test_file])
        test_iter = DataLoader(dataset, batch_size=config.batch_size, collate_fn=Batch.from_samples)
        if len(config.model_paths) == 1:
            model = Segmenter.load_model(config.model_paths[0], config)
        elif len(config.model_paths) > 1:
            model = SegmenterEnsemble.load_model(config.model_paths, config)
        else:
            raise ValueError("no model path...")

        if len(config.model_paths) > 1:
            for m in model.models:
                scores = Trainer.valid(m, test_iter)
                print("Evaluation")
                print("> Number of instances:", len(dataset.items))
                print("> Scores", scores)

        scores = Trainer.valid(model, test_iter)
        print("Evaluation")
        print("> Number of instances:", len(dataset.items))
        print("> Scores", scores)


if __name__ == '__main__':
    main()
