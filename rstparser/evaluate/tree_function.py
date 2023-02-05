from nltk import Tree


def convert2rst_tree(attach_tree):
    def helper(tree, parent_label, position):
        # labelの分解
        ns, relation = parent_label.split(':')
        left_ns, right_ns = ns.split('-')
        if ns == 'satellite-nucleus':
            left_relation = relation
            right_relation = 'Span'
        elif ns == 'nucleus-satellite':
            left_relation = 'Span'
            right_relation = relation
        elif ns == 'nucleus-nucleus':
            left_relation = right_relation = relation
        elif ns == 'dummy-dummy':
            left_relation = right_relation = relation
        else:
            print('unknown label')
            exit()

        # labelの復元
        if position == 'left':
            label = ':'.join([left_ns, left_relation])
        elif position == 'right':
            label = ':'.join([right_ns, right_relation])
        else:  # position == 'ROOT':
            label = 'ROOT'

        # 再帰的に木を構築
        if len(tree) < 2:
            children = [tree]
        else:  # len(tree) == 2:
            left_tree = helper(tree[0], tree.label(), 'left')
            right_tree = helper(tree[1], tree.label(), 'right')
            children = [left_tree, right_tree]
        return Tree(label, children)

    rst_tree = helper(attach_tree, 'dummy-dummy:dummy', 'ROOT')
    return rst_tree


def get_brackets(tree, eval_type):
    spans = []
    for position in tree.treepositions():
        subtree = tree[position]
        if isinstance(subtree, str) or isinstance(subtree, int):
            continue
        label = subtree.label()
        if label in ['ROOT', 'text']:
            continue

        edu_indices = [int(idx) for idx in subtree.leaves()]
        boundary = (edu_indices[0], edu_indices[-1])
        ns, relation = label.split(':')
        if eval_type == 'full':
            span = (boundary, ns, relation)
        elif eval_type == 'relation':
            span = (boundary, relation)
        elif eval_type == 'ns':
            span = (boundary, ns)
        elif eval_type == 'span':
            span = (boundary)
        else:
            raise ValueError('unknown eval_type')
        spans.append(span)
    return spans
