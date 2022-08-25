from nltk import Tree


def tree_division(_dataset, x2y):
    dataset = []
    for data in _dataset:
        doc_id = data['doc_id']
        tree = Tree.fromstring(data['labelled_attachment_tree'])
        edu_strings = data['tokenized_strings']
        edu_starts_sentence = data['starts_sentence']
        edu_starts_paragraph = data['starts_paragraph']
        sentence_treepositions, paragraph_treepositions = get_treepositions(
            tree, edu_starts_sentence, edu_starts_paragraph)
        positions = {'sentence': sentence_treepositions,
                     'paragraph': paragraph_treepositions}

        if x2y in ['d2s', 'd2p']:
            target = 'sentence' if x2y.endswith('s') else 'paragraph'
            tree, _ = separate_tree(tree, positions[target])
            if tree is None:
                continue

            edu_strings = get_edu_strings(tree, edu_strings)
            parent_label = None
            xxx_starts_sentence = get_starts_xxx(tree, edu_starts_sentence)
            xxx_starts_paragraph = get_starts_xxx(tree, edu_starts_paragraph)
            tree = init_edu_idx(tree)
            text_span, _ = make_text_span(edu_strings, doc_id=doc_id)
            tokenized_edu_strings = edu_strings

            dataset.append({
                'doc_id': doc_id,
                'rst_tree': None,
                'labelled_attachment_tree': tree._pformat_flat("", "()", False),
                'raw_tokenized_strings': [edu_string.split() for edu_string in tokenized_edu_strings],
                'spans': text_span,
                'starts_sentence': xxx_starts_sentence,
                'starts_paragraph': xxx_starts_paragraph,
                'parent_label': parent_label,
                'granularity_type': x2y.upper(),
            })

        elif x2y in ['p2e', 'p2s', 's2e']:
            if x2y == 'p2s':
                d2s_tree, _ = separate_tree(tree, positions['sentence'])
                d2p_tree, p2s_trees = separate_tree(d2s_tree, positions['paragraph'])
                ptree = d2p_tree
                subtrees = p2s_trees
            else:
                target = 'sentence' if x2y.startswith('s') else 'paragraph'
                ptree, subtrees = separate_tree(tree, positions[target])

            if ptree is None:
                continue

            parent_labels = get_parent_labels(ptree)
            start_offset = 0
            for subtree, parent_label in zip(subtrees, parent_labels):
                subtree_edu_strings = get_edu_strings(subtree, edu_strings)
                starts_sentence = get_starts_xxx(subtree, edu_starts_sentence)
                starts_paragraph = get_starts_xxx(subtree, edu_starts_paragraph)
                subtree = init_edu_idx(subtree)
                text_span, start_offset = make_text_span(subtree_edu_strings, start_offset, doc_id=doc_id)
                tokenized_edu_strings = subtree_edu_strings
                if len(subtree) == 1:
                    continue
                dataset.append({
                    'doc_id': doc_id,
                    'rst_tree': None,
                    'labelled_attachment_tree': subtree._pformat_flat("", "()", False),
                    'raw_tokenized_strings': [edu_string.split() for edu_string in tokenized_edu_strings],
                    'spans': text_span,
                    'starts_sentence': starts_sentence,
                    'starts_paragraph': starts_paragraph,
                    'parent_label': parent_label,
                    'granularity_type': x2y.upper(),
                })

    return dataset


def separate_tree(tree, positions):
    if positions == [()]:
        return None, []

    subtrees = []
    for tp in positions:
        subtree = tree[tp]
        subtrees.append(subtree)
        span_txt = get_span_txt(subtree)
        tree[tp] = Tree('text', [span_txt])

    return tree, subtrees


def get_span_txt(tree):
    leaves = tree.leaves()
    if len(leaves) == 1:
        return leaves[0]

    left = leaves[0].split('-')[0]
    right = leaves[-1].split('-')[-1]
    span_txt = '-'.join([left, right])
    return span_txt


def get_edu_strings(tree, edu_strings):
    xxx_strings = []
    for leave in tree.leaves():
        if len(leave.split('-')) == 2:  # span
            left, right = map(int, leave.split('-'))
            strings = []
            for edu_idx in range(left, right + 1):
                strings.append(edu_strings[edu_idx])
            string = ' '.join(strings)
        else:
            edu_idx = int(leave)
            string = edu_strings[edu_idx]

        xxx_strings.append(string)

    return xxx_strings


def make_text_span(edu_strings, starts_offset=0, doc_id=None):
    spans = []
    offset = starts_offset
    for edu_stirng in edu_strings:
        words = edu_stirng.split()
        n_words = len(words)
        if doc_id is not None:
            span = (offset, offset + n_words, doc_id)
        else:
            span = (offset, offset + n_words)
        spans.append(span)
        offset += n_words

    assert len(spans) == len(edu_strings)
    return spans, offset


def get_starts_xxx(tree, edu_starts_xxx):
    starts_xxx = []
    for leave in tree.leaves():
        if len(leave.split('-')) == 2:
            edu_idx = int(leave.split('-')[0])
        else:
            edu_idx = int(leave)

        starts_xxx.append(edu_starts_xxx[edu_idx])

    return starts_xxx


def get_parent_labels(tree):
    labels = []
    for tp in tree.treepositions('leave'):
        l = tree[tp[:-2]].label()
        labels.append(l)

    return labels


def init_edu_idx(tree):
    for idx, tp in enumerate(tree.treepositions('leave')):
        tree[tp] = str(idx)

    return tree


def get_treepositions(tree, edu_starts_sentence, edu_starts_paragraph):
    # sentence / paragraphが含むEDUのspanを獲得
    sentence_edu_spans = edu_starts_xx2spans(edu_starts_sentence)
    paragraph_edu_spans = edu_starts_xx2spans(edu_starts_paragraph)

    # sentence / paragraph のspanからpositionを求め、重複があれば分解
    _sentence_positions = spans2tree_positions(sentence_edu_spans, tree)
    sentence_positions = search_new_position([], _sentence_positions, tree)
    _paragraph_positions = spans2tree_positions(paragraph_edu_spans, tree)
    paragraph_positions = search_new_position([], _paragraph_positions, tree)

    # tupleへと変換
    sentence_positions = [tuple(tp) for tp in sentence_positions]
    paragraph_positions = [tuple(tp) for tp in paragraph_positions]
    return sentence_positions, paragraph_positions


def edu_starts_xx2spans(edu_starts_list):
    spans = []
    start = 0
    for i in range(len(list(filter(lambda x: x, edu_starts_list)))):
        if True in edu_starts_list[start + 1:]:
            _start = start + 1
            end = _start + edu_starts_list[_start:].index(True)
        else:
            end = len(edu_starts_list)

        spans.append([start, end])
        start = end

    return spans


def spans2tree_positions(spans, tree):
    positions = []
    for start, end in spans:
        position = tree.treeposition_spanning_leaves(start, end)
        if type(tree[position]) == str:
            # 単一の要素を指すときはその上のノード
            position = position[:-2]

        positions.append(position)

    return positions


def has_duplication(x, others):
    """
    position:xとxを含まないそのほかのposition:othersを受け取り、
    それらの中で、重複(部分木に頂点が含まれる)ものが存在すればTrue
    """

    def helper(x, y):
        if len(x) == 0:
            return True  # ROOT
        for _x, _y in zip(x, y):
            if _x != _y:
                return False

        return True

    return any([helper(x, y) for y in others])


def search_new_position(position, others, tree):
    """
    positionの部分木に含まれるnodeがothersにあれば
    再帰的に分割を行い、排反となるpositionのリストを返す。
    """
    if tree[position].height() <= 3:  # 一番下まで降らないようにする
        return [position]  # 特別終了条件

    _others = filter(lambda x: len(x) > len(position), others)
    if not has_duplication(position, _others):
        return [position]  # 終了条件
    else:
        positions = []
        for i in range(len(tree[position])):
            positions += search_new_position(position + [i], others, tree)
        return positions
