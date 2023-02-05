from collections import defaultdict

import torch
import torch.nn as nn
from nltk import Tree

from rstparser.dataset.data_loader import Batch
from rstparser.dataset.merge_file import Doc
from rstparser.networks.ensemble import EnsembleParser
from rstparser.networks.parser import SpanBasedParser


class HierarchicalParser(nn.Module):
    def __init__(self, parsers, hierarchical_type, device):
        super(HierarchicalParser, self).__init__()
        self.d2e_parser = parsers.get('d2e', None)
        self.d2p_parser = parsers.get('d2p', None)
        self.d2s_parser = parsers.get('d2s', None)
        self.p2s_parser = parsers.get('p2s', None)
        self.p2e_parser = parsers.get('p2e', None)
        self.s2e_parser = parsers.get('s2e', None)

        assert hierarchical_type in ['d2e', 'd2s2e', 'd2p2s2e']
        self.hierarchical_type = hierarchical_type
        self.device = device

    @staticmethod
    def load_model(model_paths, config):
        # load model and classify hierarchical type
        hierarchical_models = defaultdict(list)
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        for model_path in model_paths:
            model = SpanBasedParser.load_model(model_path, config)
            h_type = model.hierarchical_type
            assert h_type in ['d2e', 'd2s', 'd2p', 'p2e', 'p2s', 's2e']
            hierarchical_models[h_type].append(model)

        parsers = {}
        for h_type, models in hierarchical_models.items():
            print('{}'.format(h_type), end='\t')
            if len(models) == 0:
                parsers[h_type] = None
            if len(models) == 1:
                print('single')
                parsers[h_type] = models[0]
            if len(models) > 1:
                print('ensemble')
                parsers[h_type] = EnsembleParser(models, device)

        return HierarchicalParser(parsers, config.hierarchical_type, device)

    def parse(self, doc) -> Tree:
        output = self(doc)
        tree = output['tree'][0]
        return tree

    def forward(self, batch):
        if isinstance(batch, Batch):
            doc = Doc.from_batch(batch)
        else:
            doc = batch

        if self.hierarchical_type == 'd2p2s2e':
            batch = doc.to_batch(parent_label=None, x2y='d2p', index=0)
            d2p_tree = self.d2p_parser(batch)['tree'][0]

            p2s_trees = []
            for p_idx, parent_label in enumerate(get_leaf_labels(d2p_tree)):
                batch = doc.to_batch(parent_label=parent_label, x2y='p2s', index=p_idx)
                p2s_trees.append(self.p2s_parser(batch)['tree'][0])
            d2s_tree = joint_tree(d2p_tree, p2s_trees)

            s2e_trees = []
            for p_idx, parent_label in enumerate(get_leaf_labels(d2s_tree)):
                batch = doc.to_batch(parent_label=parent_label, x2y='s2e', index=p_idx)
                s2e_trees.append(self.s2e_parser(batch)['tree'][0])
            d2e_tree = joint_tree(d2s_tree, s2e_trees)

        elif self.hierarchical_type == 'd2p2e':
            batch = doc.to_batch(parent_label=None, x2y='d2p', index=0)
            d2p_tree = self.d2p_parser(batch)['tree'][0]

            p2e_trees = []
            for p_idx, parent_label in enumerate(get_leaf_labels(d2p_tree)):
                batch = doc.to_batch(parent_label=parent_label, x2y='p2e', index=p_idx)
                p2e_trees.append(self.p2e_parser(batch)['tree'][0])
            d2e_tree = joint_tree(d2p_tree, p2e_trees)

        elif self.hierarchical_type == 'd2s2e':
            batch = doc.to_batch(parent_label=None, x2y='d2s', index=0)
            d2s_tree = self.d2s_parser(batch)['tree'][0]

            s2e_trees = []
            for p_idx, parent_label in enumerate(get_leaf_labels(d2s_tree)):
                batch = doc.to_batch(parent_label=parent_label, x2y='s2e', index=p_idx)
                s2e_trees.append(self.s2e_parser(batch)['tree'][0])
            d2e_tree = joint_tree(d2s_tree, s2e_trees)

        elif self.hierarchical_type == 'd2e':
            batch = doc.to_batch()
            d2e_tree = self.d2e_parser(batch)['tree'][0]
            d2e_tree = Tree.fromstring(d2e_tree)
        else:
            raise ValueError('Unknown hierarchical type')

        return {'tree': [d2e_tree]}


def get_leaf_labels(tree):
    if not isinstance(tree, Tree):
        tree = Tree.fromstring(tree)
    if tree.height() <= 2:  # single element makes a tree
        return [None]
    else:
        return [tree[p[:-2]].label() for p in tree.treepositions('leaves')]


def joint_tree(parent_tree, child_trees):
    if not isinstance(parent_tree, Tree):
        parent_tree = Tree.fromstring(parent_tree)
    if not isinstance(child_trees[0], Tree):
        child_trees = [Tree.fromstring(tree) for tree in child_trees]
    joint_tree = parent_tree.copy(True)
    # Consists of a single element (if an assignment results in an error)
    if len(joint_tree) == 1:
        joint_tree = Tree('D-ROOT', [joint_tree])
    # Replace leave in joint_tree with child_tree
    for leave_idx, leave_position in enumerate(joint_tree.treepositions('leaves')):
        leave_position = leave_position[:-1]  # (nucleus:relation (text N))
        joint_tree[leave_position] = child_trees[leave_idx]
    # Remove D-ROOT
    if len(joint_tree) == 1:
        joint_tree = joint_tree[0]
    # Reassign the EDU index
    for leave_idx, leave_position in enumerate(joint_tree.treepositions('leaves')):
        joint_tree[leave_position] = leave_idx
    return joint_tree
