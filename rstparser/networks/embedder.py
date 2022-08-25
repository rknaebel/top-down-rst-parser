import sys
from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from rstparser.networks.layers import BiLSTM, SelectiveGate


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout, padding_idx=None):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices):
        if isinstance(indices, tuple):
            indices = indices[0]
            # indices[1]: lengths
        embeddings = self.embed(indices)
        return self.dropout(embeddings)

    def get_embed_size(self):
        return self.embed.weight.size(1)


class BertEmbeddingModel(nn.Module):
    def __init__(self, bert_model, dropout, device, batch_size=8, last_hidden_only=True):
        super(BertEmbeddingModel, self).__init__()
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        if bert_model.startswith('roberta'):
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.device = device
        self.model = AutoModel.from_pretrained(bert_model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.last_hidden_only = last_hidden_only
        self.simple_map = {
            "''": '"',
            "``": '"',
            "-LRB-": "(",
            "-RRB-": ")",
            "-LCB-": "{",
            "-RCB-": "}",
            "n't": "not"
        }
        self.cache = dict()

    @staticmethod
    def build_model(config):
        dropout = getattr(config, 'dropout', 0.4)
        batch_size = getattr(config, 'embed_batch_size', 8)
        bert_model = getattr(config, 'bert_model', 'bert-base-cased')
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        embedder = BertEmbeddingModel(bert_model, dropout, device, batch_size, last_hidden_only=False)
        embedder.to(device)
        return embedder

    def forward(self, raw_text, starts_sentence):
        assert len(raw_text) == len(starts_sentence)
        all_sentences = []
        for edus, edu_starts_sentence in zip(raw_text, starts_sentence):
            ends_sentence = edu_starts_sentence[1:] + [True]
            sentences, sentence = [], []
            for edu_words, end_of_sentence in zip(edus, ends_sentence):
                sentence.extend(edu_words)
                if end_of_sentence:
                    sentences.append(sentence)
                    sentence = []
            all_sentences.extend(sentences)

        # Run Embedder
        sentence_embeddings = []
        for min_batch in self.batch_iter(all_sentences, self.batch_size):
            sentence_embeddings.extend(self._forward(min_batch))

        # Sentence embeddings -> EDU embeddings
        sentence_idx = 0
        batch_edu_embeddings = []
        for edus, edu_starts_sentence in zip(raw_text, starts_sentence):
            ends_sentence = edu_starts_sentence[1:] + [True]
            edu_offset = 0
            edu_embeddings = []
            for edu_words, end_of_sentence in zip(edus, ends_sentence):
                edu_length = len(edu_words)
                edu_embedding = sentence_embeddings[sentence_idx][edu_offset: edu_offset + edu_length]
                edu_embeddings.append(edu_embedding)

                edu_offset += edu_length
                if end_of_sentence:
                    sentence_idx += 1
                    edu_offset = 0

            # edu_embeddings: Num_edus, Num_words, embedding_size
            edu_embeddings = pad_sequence(edu_embeddings, batch_first=True, padding_value=0)
            max_num_words = max(len(e) for s in raw_text for e in s)
            diff = max_num_words - edu_embeddings.size(1)
            edu_embeddings = torch.nn.functional.pad(edu_embeddings, (0, 0, 0, diff))
            batch_edu_embeddings.append(edu_embeddings)

        embeddings = pad_sequence(batch_edu_embeddings, batch_first=True, padding_value=0)
        return self.dropout(embeddings)

    def convert_tokens(self, sentences):
        return [[self.simple_map.get(t, t) for t in sent] for sent in sentences]

    def compute_subtoken_lengths(self, tokens):
        return [[len(self.tokenizer.tokenize(t)) for t in sent] for sent in tokens]

    def _forward(self, sentences: List[List[str]]):
        tokens = self.convert_tokens(sentences)
        lengths = self.compute_subtoken_lengths(tokens)
        inputs = self.tokenizer(tokens, padding=True, return_tensors='pt', is_split_into_words=True)
        inputs.to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)

        if self.last_hidden_only:
            hidden_state = outputs.hidden_states[-2]
        else:
            hidden_state = torch.cat(outputs.hidden_states[-4:], dim=-1)
        embeddings = [torch.zeros((len(s), hidden_state.shape[-1]), device=self.device) for s in tokens]
        for sent_i, _ in enumerate(inputs['input_ids']):
            e_i = 0
            len_left = 1
            for length in lengths[sent_i]:
                embeddings[sent_i][e_i] = hidden_state[sent_i][len_left]
                len_left += length
                e_i += 1
        return embeddings

    @staticmethod
    def batch_iter(iterable, batch_size=1):
        length = len(iterable)
        for offset in range(0, length, batch_size):
            yield iterable[offset:min(offset + batch_size, length)]

    def get_embed_size(self):
        if self.last_hidden_only:
            return self.model.config.hidden_size
        else:
            return self.model.config.hidden_size * 4


class TextEmbedder(nn.Module):
    def __init__(self, word_embedder, hidden_size, dropout, device, use_gate=False):
        super(TextEmbedder, self).__init__()
        self.word_embedder = word_embedder
        self.use_gate = use_gate
        self.device = device
        if use_gate:
            embed_size = word_embedder.get_embed_size()
            self.gate_lstm = SelectiveGate(BiLSTM(embed_size, hidden_size, dropout))

    @staticmethod
    def build_model(config):
        word_embedder = BertEmbeddingModel.build_model(config)
        hidden_size = getattr(config, 'hidden', 250)
        dropout = getattr(config, 'dropout', 0.4)
        use_gate = getattr(config, 'gate_embed', True)
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        text_embedder = TextEmbedder(word_embedder, hidden_size, dropout, device, use_gate)
        text_embedder.to(device)
        return text_embedder

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, edu_lengths, word_lengths, raw_inputs, starts_sentence):
        # edu_lengts: (batch, )
        edu_lengths = edu_lengths.to(self.device)
        # word_lengths: (batch, num_edus), number of words that make up each edu
        word_lengths = word_lengths.to(self.device)

        try:
            # word_embeddings: (batch, num_edus, num_words, embed_dim)
            word_embeddings = self.word_embedder(raw_inputs, starts_sentence)
        except RuntimeError as e:
            # TODO repair too long inputs
            # where do they come from - seems only single one
            sys.stderr.write(f'>> Error: {e}')
            raise RuntimeError

        if self.use_gate:
            # batch expansion
            edu_embeddings = []
            for _embeddings, num_edus, lengths in zip(word_embeddings, edu_lengths, word_lengths):
                gated_rnn_outputs, _ = self.gate_lstm(_embeddings[:num_edus], lengths[:num_edus])
                gated_sum_embeddings = torch.sum(gated_rnn_outputs, dim=1)
                gated_mean_embeddings = gated_sum_embeddings / lengths[:num_edus].unsqueeze(-1).float()
                edu_embeddings.append(gated_mean_embeddings)
            # rebuild
            edu_embeddings = torch.nn.utils.rnn.pad_sequence(edu_embeddings, batch_first=True)
        else:
            edu_embeddings = torch.sum(word_embeddings, dim=2) / word_lengths.unsqueeze(-1).float()

        return edu_embeddings

    def get_embed_size(self):
        if self.use_gate:
            embed_size = self.gate_lstm.lstm.hidden_size
            if self.gate_lstm.lstm.bidirectional:
                embed_size *= 2
        else:
            embed_size = self.word_embedder.get_embed_size()
        return embed_size
