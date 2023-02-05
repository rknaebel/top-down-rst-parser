import functools

import torch
import torch.nn as nn

import rstparser.dataset.trees as rsttree
from rstparser.dataset.data_loader import Batch
from rstparser.networks.embedder import TextEmbedder, Embeddings
from rstparser.networks.layers import BiLSTM, FeedForward, DeepBiAffine
from rstparser.trainer.checkpointer import Checkpointer


class SpanBasedParser(nn.Module):
    def __init__(self, text_embedder: TextEmbedder, hidden_size, margin, dropout,
                 ns_vocab, relation_vocab, device, hierarchical_type,
                 label_type, use_parent_label, use_hard_boundary):
        super(SpanBasedParser, self).__init__()
        self.hidden_size = hidden_size
        self.margin = margin
        self.dropout = dropout
        self.device = device
        self.hierarchical_type = hierarchical_type
        self.label_type = label_type
        self.use_parent_label = use_parent_label
        self.use_hard_boundary = use_hard_boundary

        self.ns_vocab = ns_vocab
        self.rela_vocab = relation_vocab

        # Embeddings
        """ Text embedder """
        self.text_embedder: TextEmbedder = text_embedder
        embed_size = self.text_embedder.get_embed_size()

        """ Parent label embedder """
        self.label_embed_size = 16
        self.ns_embedder = Embeddings(len(ns_vocab), self.label_embed_size,
                                      0.0, padding_idx=ns_vocab.stoi['<pad>'])
        self.rela_embedder = Embeddings(len(relation_vocab), self.label_embed_size,
                                        0.0, padding_idx=relation_vocab.stoi['<pad>'])
        """ Bound embedder """
        self.bound_embed_size = 16
        self.bound_embedder = Embeddings(16, self.bound_embed_size, 0.0)

        # BiLSTM
        self.bilstm = BiLSTM(embed_size, self.hidden_size, self.dropout)
        self.edge_pad = nn.ZeroPad2d((0, 0, 1, 1))
        # Scoring functions
        """ Split Scoring """
        self.span_embed_size = self.hidden_size * 2 + self.bound_embed_size
        self.f_split = DeepBiAffine(self.span_embed_size, self.dropout)
        """ Label Scoring """
        feature_embed_size = self.span_embed_size * 4
        if use_parent_label:
            feature_embed_size += self.label_embed_size * 2
        self.f_ns = FeedForward(feature_embed_size, [self.hidden_size], len(ns_vocab), self.dropout)
        self.f_rela = FeedForward(feature_embed_size, [self.hidden_size], len(relation_vocab), self.dropout)

    def freeze(self):
        self.text_embedder.freeze()
        self.bilstm.freeze()

    @staticmethod
    def build_model(config, ns_vocab, rel_vocab):
        embedder = TextEmbedder.build_model(config)
        hidden = config.hidden
        margin = config.margin
        dropout = config.dropout
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        hierarchical_type = config.hierarchical_type
        label_type = getattr(config, 'label_type', 'full')
        use_parent_label = getattr(config, 'parent_label_embed', True)
        use_hard_boundary = getattr(config, 'use_hard_boundary', False)
        model = SpanBasedParser(embedder, hidden, margin, dropout, ns_vocab, rel_vocab, device,
                                hierarchical_type, label_type, use_parent_label, use_hard_boundary)
        model.to(device)
        return model

    @staticmethod
    def load_model(model_path, config):
        print('load model: {}'.format(model_path))
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        model_state = Checkpointer.restore(model_path, device=device)
        model_config = model_state['config']
        model_config.cpu = config.cpu
        model_config.use_hard_boundary = config.use_hard_boundary
        model_param = model_state['model']
        ns_vocab = model_state['vocab']['ns']
        rel_vocab = model_state['vocab']['rel']
        model = SpanBasedParser.build_model(model_config, ns_vocab, rel_vocab)
        model.load_state_dict(model_param, strict=False)
        model.eval()
        return model

    def parse(self, doc, parent_label=None, index=0) -> str:
        batch = doc.to_batch(parent_label=parent_label, x2y=self.hierarchical_type, index=index)
        output = self.forward(batch)
        tree = output['tree'][0]
        return tree

    def forward(self, batch):
        rnn_outputs = self.embed(batch)
        losses = []
        pred_trees = []
        for i in range(len(batch)):
            tree, loss = self.greedy_tree(
                rnn_outputs[i],
                batch.starts_sentence[i],
                batch.starts_paragraph[i],
                batch.parent_label[i],
                batch.tree[i].convert() if batch.tree and self.training else None)
            losses.append(loss)
            pred_trees.append(tree.convert().linearize())

        loss = torch.mean(torch.stack(losses))

        return {
            'loss': loss,
            'tree': pred_trees,
        }

    def embed(self, batch: Batch):
        edu_embeddings = self.text_embedder(batch.edu_len, batch.words_len, batch.elmo_word, batch.starts_sentence)
        lstm_outputs = self.bilstm(edu_embeddings, batch.edu_len)
        # lstm_outputs: (batch, num_edus, hidden)
        lstm_outputs = [output[:l] for output, l in zip(lstm_outputs, batch.word[1])]
        return lstm_outputs

    def greedy_tree(self, rnn_output, starts_sentence, starts_paragraph, parent_label, gold_tree=None):
        rnn_output = self.edge_pad(rnn_output)
        sentence_length = len(rnn_output) - 2
        self.rnn_output = rnn_output
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.gold_tree = gold_tree

        @functools.lru_cache(maxsize=None)
        def helper(left, right, parent_label=None):
            assert 0 <= left < right <= sentence_length
            if right - left == 1:  # 終了条件
                tag, word = 'text', str(left)
                tree = rsttree.LeafParseNode(left, tag, word)
                return tree, torch.zeros(1, device=self.device).squeeze()

            split_scores = self.get_split_scores(left, right)
            try:
                split, split_loss = self.predict_split(split_scores, left, right)
            except Exception as e:
                print("\nERROR at", gold_tree, e, "\n", split_scores, left, right)
                exit(1)

            feature = self.get_feature_embedding(left, right, split, parent_label)
            ns_label_scores = self.f_ns(feature)
            rela_label_scores = self.f_rela(feature)
            ns, ns_loss = self.predict_label(ns_label_scores, left, right,
                                             sentence_length, self.ns_vocab, 0)
            rela, rela_loss = self.predict_label(rela_label_scores, left, right,
                                                 sentence_length, self.rela_vocab, 1)

            left_trees, left_loss = helper(left, split, (ns, rela))
            right_trees, right_loss = helper(split, right, (ns, rela))
            children = rsttree.InternalParseNode((':'.join((ns, rela)),), [left_trees, right_trees])

            if self.label_type == 'skelton':
                loss = split_loss + left_loss + right_loss
            elif self.label_type == 'ns':
                loss = ns_loss + split_loss + left_loss + right_loss
            elif self.label_type == 'full':
                loss = ns_loss + rela_loss + split_loss + left_loss + right_loss

            return children, loss

        pred_tree, loss = helper(0, sentence_length, parent_label)
        if self.training:
            assert gold_tree.convert().linearize() == pred_tree.convert().linearize(), f"Gold:{gold_tree.convert().linearize()} PRED:{pred_tree.convert().linearize()}"
        return pred_tree, loss

    def get_split_scores(self, left, right):
        left_encodings = []
        right_encodings = []
        for k in range(left + 1, right):
            left_encodings.append(self.get_span_embedding(left, k))
            right_encodings.append(self.get_span_embedding(k, right))

        left_encodings = torch.stack(left_encodings)
        right_encodings = torch.stack(right_encodings)
        split_scores = self.f_split(left_encodings, right_encodings)
        split_scores = split_scores.view(len(left_encodings))

        if self.use_hard_boundary:
            paragraph_split = [not f for f in self.starts_paragraph[left + 1:right]]
            sentence_split = [not f for f in self.starts_sentence[left + 1:right]]
            min_value = min(split_scores) - 10.0
            if not all(paragraph_split):
                split_scores[paragraph_split] = min_value
            if not all(sentence_split):
                split_scores[sentence_split] = min_value

        return split_scores

    def get_span_embedding(self, left, right):
        if left == right:
            return torch.zeros([self.span_embed_size], device=self.device)

        forward = (
                self.rnn_output[right][:self.hidden_size] -
                self.rnn_output[left][:self.hidden_size])
        backward = (
                self.rnn_output[left + 1][self.hidden_size:] -
                self.rnn_output[right + 1][self.hidden_size:])

        bound_embedding = self.get_boundary_embedding(left, right)
        span_embedding = torch.cat([forward, backward, bound_embedding])
        return span_embedding

    def get_feature_embedding(self, left, right, split, parent_label):
        left_span = self.get_span_embedding(left, split)
        right_span = self.get_span_embedding(split, right)
        label_embedding = self.get_label_embedding(parent_label)

        N = len(self.rnn_output) - 2
        out_left_span = self.get_span_embedding(0, left)
        out_right_span = self.get_span_embedding(right, N)
        if self.use_parent_label:
            feature = torch.cat([left_span, right_span, label_embedding,
                                 out_left_span, out_right_span], dim=0)
        else:
            feature = torch.cat([left_span, right_span,
                                 out_left_span, out_right_span], dim=0)
        return feature

    def get_boundary_embedding(self, left, right):
        is_start_sentence = self.starts_sentence[left]
        is_start_paragraph = self.starts_paragraph[left]
        cross_sentence = any(self.starts_sentence[left + 1: right])
        cross_paragraph = any(self.starts_paragraph[left + 1: right])
        bound = int(is_start_sentence * 1 + is_start_paragraph * 2 + cross_sentence * 4 + cross_paragraph * 8)
        bound = torch.tensor(bound, dtype=torch.long, device=self.device)
        bound_embedding = self.bound_embedder(bound)
        return bound_embedding

    def get_label_embedding(self, label):
        if label is None:
            label = '<pad>:<pad>'
        if isinstance(label, str):
            label = label.split(':')  # split 'NS:Relation' into ('NS', 'Relation')

        if self.label_type == 'skelton':
            label = ['<pad>', '<pad>']
        elif self.label_type == 'ns':
            label = [label[0], '<pad>']
        elif self.label_type == 'full':
            pass

        ns, relation = label
        ns_idx = torch.tensor(self.ns_vocab.stoi[ns], dtype=torch.long, device=self.device)
        ns_embedding = self.ns_embedder(ns_idx)
        rela_idx = torch.tensor(self.rela_vocab.stoi[relation], dtype=torch.long, device=self.device)
        rela_embedding = self.rela_embedder(rela_idx)

        label_embedding = torch.cat([ns_embedding, rela_embedding], dim=-1)
        return label_embedding

    def augment(self, scores, oracle_index):
        assert len(scores.size()) == 1
        increment = torch.ones_like(scores) + self.margin
        increment[oracle_index] = 0
        return scores + increment

    def predict_split(self, split_scores, left, right):
        if self.training:
            oracle_split = min(self.gold_tree.oracle_splits(left, right))
            oracle_split_index = oracle_split - (left + 1)
            split_scores = self.augment(split_scores, oracle_split_index)

        split_scores_np = split_scores.data.cpu().numpy()
        argmax_split_index = int(split_scores_np.argmax())
        argmax_split = argmax_split_index + (left + 1)

        if self.training:
            split = oracle_split
            split_loss = (
                split_scores[argmax_split_index] - split_scores[oracle_split_index]
                if argmax_split != oracle_split else torch.zeros(1, device=self.device).squeeze())
        else:
            split = argmax_split
            split_loss = split_scores[argmax_split_index]

        return split, split_loss

    def predict_label(self, label_scores, left, right, sentence_length, vocab, label_idx):
        if self.training:
            oracle_label = self.gold_tree.oracle_label(left, right)[0].split(':')[label_idx]
            oracle_label_index = vocab.stoi[oracle_label]
            label_scores = self.augment(label_scores, oracle_label_index)

        label_scores_np = label_scores.data.cpu().numpy()
        argmax_label_index = int(
            label_scores_np.argmax() if right - left < sentence_length else
            label_scores_np[1:].argmax() + 1)
        argmax_label = vocab.itos[argmax_label_index]

        if self.training:
            label = oracle_label
            label_loss = (
                label_scores[argmax_label_index] - label_scores[oracle_label_index]
                if argmax_label != oracle_label else torch.zeros(1, device=self.device).squeeze())
        else:
            label = argmax_label
            label_loss = label_scores[argmax_label_index]

        return label, label_loss
