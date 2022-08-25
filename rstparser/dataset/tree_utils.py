import logging
import re

import click
from nltk.tree import ParentedTree, Tree

_ptb_paren_mapping = {'(': r"-LRB-",
                      ')': r"-RRB-",
                      '[': r"-LSB-",
                      ']': r"-RSB-",
                      '{': r"-LCB-",
                      '}': r"-RCB-"}
_reverse_ptb_paren_mapping = {bracket_replacement: bracket_type
                              for bracket_type, bracket_replacement
                              in _ptb_paren_mapping.items()}

RELATION_TABLE = {
    "ROOT": "ROOT",
    "span": "Span",
    "attribution": "Attribution",
    "attribution-negative": "Attribution",
    "background": "Background",
    "circumstance": "Background",
    "cause": "Cause",
    "result": "Cause",
    "cause-result": "Cause",
    "consequence": "Cause",
    "comparison": "Comparison",
    "preference": "Comparison",
    "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition",
    "hypothetical": "Condition",
    "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast",
    "concession": "Contrast",
    "antithesis": "Contrast",
    "elaboration-additional": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-process-step": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "example": "Elaboration",
    "definition": "Elaboration",
    "enablement": "Enablement",
    "purpose": "Enablement",
    "evaluation": "Evaluation",
    "interpretation": "Evaluation",
    "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation",
    "explanation-argumentative": "Explanation",
    "reason": "Explanation",
    "list": "Joint",
    "disjunction": "Joint",
    "manner": "Manner-Means",
    "means": "Manner-Means",
    "problem-solution": "Topic-Comment",
    "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment",
    "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment",
    "rhetorical-question": "Topic-Comment",
    "summary": "Summary",
    "restatement": "Summary",
    "temporal-before": "Temporal",
    "temporal-after": "Temporal",
    "temporal-same-time": "Temporal",
    "sequence": "Temporal",
    "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change",
    "topic-drift": "Topic-Change",
    "textualorganization": "Textual-organization",
    "same-unit": "Same-unit"
}


def fix_rst_edu(edu_path_basename, edu):
    edu = re.sub(r">\s*", r'', edu).replace("&amp;", '&')
    edu = re.sub(r"---", r"--", edu)
    edu = edu.replace(". . .", "...")

    # annoying edge cases that we fix manually
    if edu_path_basename == 'file1.edus':
        edu = edu.replace('founded by',
                          "founded by his grandfather.")
    elif (edu_path_basename == "wsj_0660.out.edus" or
          edu_path_basename == "wsj_1368.out.edus" or
          edu_path_basename == "wsj_1371.out.edus"):
        edu = edu.replace("S.p. A.", "S.p.A.")
    elif edu_path_basename == "wsj_1329.out.edus":
        edu = edu.replace("G.m.b. H.", "G.m.b.H.")
    elif edu_path_basename == "wsj_1367.out.edus":
        edu = edu.replace("-- that turban --",
                          "-- that turban")
    elif edu_path_basename == "wsj_1377.out.edus":
        edu = edu.replace("Part of a Series",
                          "Part of a Series }")
    elif edu_path_basename == "wsj_1974.out.edus":
        edu = edu.replace(r"5/ 16", r"5/16")
    elif edu_path_basename == "file2.edus":
        edu = edu.replace("read it into the record,",
                          "read it into the record.")
    elif edu_path_basename == "file3.edus":
        edu = edu.replace("about $to $", "about $2 to $4")
    elif edu_path_basename == "file5.edus":
        # There is a PTB error in wsj_2172.mrg:
        # The word "analysts" is missing from the parse.
        # It's gone without a trace :-/
        edu = edu.replace("panic among analysts",
                          "panic among")
        edu = edu.replace("his bid Oct. 17", "his bid Oct. 5")
        edu = edu.replace("his bid on Oct. 17",
                          "his bid on Oct. 5")
        edu = edu.replace("to commit $billion,",
                          "to commit $3 billion,")
        edu = edu.replace("received $million in fees",
                          "received $8 million in fees")
        edu = edu.replace("`` in light", '"in light')
        edu = edu.replace("3.00 a share", "2 a share")
        edu = edu.replace(" the Deal.", " the Deal.'")
        edu = edu.replace("' Why doesn't", "Why doesn't")
    elif edu_path_basename == "wsj_1331.out.edus":
        edu = edu.replace("`S", "'S")
    elif edu_path_basename == "wsj_1373.out.edus":
        edu = edu.replace("... An N.V.", "An N.V.")
        edu = edu.replace("features.", "features....")
    elif edu_path_basename == "wsj_1123.out.edus":
        edu = edu.replace('" Reuben', 'Reuben')
        edu = edu.replace("subscribe to.", 'subscribe to."')
    elif edu_path_basename == "wsj_2317.out.edus":
        edu = edu.replace(". The lower", "The lower")
        edu = edu.replace("$4 million", "$4 million.")
    elif edu_path_basename == 'wsj_1376.out.edus':
        edu = edu.replace("Elizabeth.", 'Elizabeth.\'"')
        edu = edu.replace('\'" In', "In")
    elif edu_path_basename == "wsj_1105.out.edus":
        # PTB error: a sentence starts with an end quote.
        # For simplicity, we'll just make the
        # EDU string look like the PTB sentence.
        edu = edu.replace("By lowering prices",
                          '"By lowering prices')
        edu = edu.replace(' 70% off."', ' 70% off.')
    elif edu_path_basename == 'wsj_1125.out.edus':
        # PTB error: a sentence ends with an start quote.
        edu = edu.replace("developer.", 'developer."')
        edu = edu.replace('"So developers', 'So developers')
    elif edu_path_basename == "wsj_1158.out.edus":
        edu = re.sub(r"\s*\-$", r'', edu)
        # PTB error: a sentence starts with an end quote.
        edu = edu.replace(' virtues."', " virtues.")
        edu = edu.replace("So much for", '"So much for')
    elif edu_path_basename == "wsj_0632.out.edus":
        # PTB error: a sentence starts with an end quote.
        edu = edu.replace(" individual.", ' individual."')
        edu = edu.replace('"If there ', "If there ")
    elif edu_path_basename == "wsj_2386.out.edus":
        # PTB error: a sentence starts with an end quote.
        edu = edu.replace('lenders."', 'lenders.')
        edu = edu.replace('Mr. P', '"Mr. P')
    elif edu_path_basename == 'wsj_1128.out.edus':
        # PTB error: a sentence ends with an start quote.
        edu = edu.replace("it down.", 'it down."')
        edu = edu.replace('"It\'s a real"', "It's a real")
    elif edu_path_basename == "wsj_1323.out.edus":
        # PTB error (or at least a very unusual edge case):
        # "--" ends a sentence.
        edu = edu.replace("-- damn!", "damn!")
        edu = edu.replace("from the hook", "from the hook --")
    elif edu_path_basename == "wsj_2303.out.edus":
        # PTB error: a sentence ends with an start quote.
        edu = edu.replace("Simpson in an interview.",
                          'Simpson in an interview."')
        edu = edu.replace('"Hooker\'s', 'Hooker\'s')
    # wsj_2343.out.edus also has an error that can't be easily
    # TODO fix: and EDU spans 2 sentences, ("to analyze what...").
    return edu


def convert_rst_tree(rst_tree_str):
    rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
    rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
    tree = ParentedTree.fromstring(rst_tree_str)
    reformat_rst_tree(tree)
    tree = binarize(tree)
    tree = re_categorize(tree)
    return convert2labelled_attachment_tree(tree)


def fix_rst_treebank_tree_str(rst_tree_str):
    """
    Fix errors in some gold standard RST trees.
    This function removes some unexplained comments in two files
    that cannot be parsed.
      - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2353.out.dis
      - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2367.out.dis
    """
    return re.sub(r'\)//TT_ERR', ')', rst_tree_str)


def convert_parens_in_rst_tree_str(rst_tree_str):
    """
    Convert parentheses in RST trees to match those in PTB trees.
    This function converts any brackets and parentheses in the EDUs of
    the RST discourse treebank to look like Penn Treebank tokens (e.g.,
    -LRB-), so that the NLTK tree API doesn't crash when trying to read
    in the RST trees.
    """
    for bracket_type, bracket_replacement in _ptb_paren_mapping.items():
        rst_tree_str = re.sub(f"(_![^_(?=!)]*)\\{bracket_type}([^_(?=!)]*_!)",
                              f"\\g<1>{bracket_replacement}\\g<2>",
                              rst_tree_str)
    return rst_tree_str


def _delete_span_leaf_nodes(tree):
    """Delete span leaf nodes."""
    subtrees = []
    subtrees.extend([s for s in tree.subtrees()
                     if s != tree and
                     (s.label() == 'span' or s.label() == 'leaf')])

    if len(subtrees) > 0:
        parent = subtrees[0].parent()
        parent.remove(subtrees[0])
        _delete_span_leaf_nodes(tree)


def _move_rel2par(tree):
    """Move the "rel2par" node."""
    subtrees = []
    subtrees.extend(
        [s for s in tree.subtrees() if s != tree and (s.label() == 'rel2par')])

    if subtrees:
        # there should only be one word describing the rel2par
        relation = ' '.join(subtrees[0].leaves())
        parent = subtrees[0].parent()

        # rename the parent node
        parent.set_label(f"{parent.label()}:{relation}".lower())

        # and then delete the rel2par node
        parent.remove(subtrees[0])
        _move_rel2par(tree)


def _replace_edu_strings(input_tree):
    """Replace EDU strings (i.e., the leaves) with indices."""
    edu_index = 0
    for subtree in input_tree.subtrees():
        if isinstance(subtree[0], str):
            subtree.clear()
            subtree.append(edu_index)
            edu_index += 1


def reformat_rst_tree(input_tree):
    """Reformat RST tree to make it look more like a PTB tree."""
    logging.debug(f"Reformatting {input_tree._pformat_flat('', '()', False)}")

    # 1. rename the top node
    input_tree.set_label("ROOT")

    # 2. delete all of the span and leaf nodes (they seem to be just for
    # book keeping)
    _delete_span_leaf_nodes(input_tree)

    # 3. move the rel2par label up to be attached to the Nucleus/Satellite
    # node
    _move_rel2par(input_tree)

    edus = input_tree.leaves()

    # 4. replace EDU strings with indices
    _replace_edu_strings(input_tree)

    logging.debug(f"Reformatted: {input_tree._pformat_flat('', '()', False)}")

    return edus, input_tree


def binarize(tree):
    if len(tree) == 1:
        # End of recursion
        return tree
    if len(tree) == 2:
        # Binary structure
        left_tree = binarize(tree[0])
        right_tree = binarize(tree[1])
    else:
        # Non-Binary structure
        labels = [tree[i].label() for i in range(len(tree))]
        is_polynuclear = all(map(lambda x: x == labels[0], labels))
        if is_polynuclear:
            # Polynuclear relation label such as:
            # same-unit, list, etc...
            # -> convert to right heavy structure
            left_tree = binarize(tree[0])
            right_tree = binarize(
                Tree(tree[0].label(), [tree[i] for i in range(1, len(tree))]))
        else:
            # Non Binary structure without Polynuclear label
            # S/N/S -> left heavy
            left_tree = binarize(Tree('nucleus:span', [tree[0], tree[1]]))
            right_tree = binarize(tree[2])

    return Tree(tree.label(), [left_tree, right_tree])


def re_categorize(rst_tree):
    # Check for RST Tree
    assert rst_tree.label() == 'ROOT'
    for position in rst_tree.treepositions():
        if not isinstance(rst_tree[position], Tree):
            continue
        if rst_tree[position].label() == 'text':
            continue
        if len(rst_tree[position]) < 2:
            continue

        sub_tree = rst_tree[position]

        # extract label
        l_ns, l_relation = sub_tree[0].label().split(':')
        r_ns, r_relation = sub_tree[1].label().split(':')

        # def _re_categorize(relation):
        #     # suffixを取り除く
        #     while relation[-2:] in ['-s', '-e', '-n']:
        #         relation = relation[:-2]
        #     return RELATION_TABLE[relation]
        #
        # # relation読み替える
        # l_relation = _re_categorize(l_relation)
        # r_relation = _re_categorize(r_relation)

        # Fixed annotation errors
        if l_ns == r_ns == 'nucleus':
            # In the case of N-N, the relation is equal for both sides, but only one case is an exception
            if l_relation != r_relation:
                # l_relation: Cause, r_relation: Span
                r_relation = 'Cause'
            assert l_relation == r_relation

        # Grant new relation
        rst_tree[position][0].set_label(':'.join([l_ns, l_relation]))
        rst_tree[position][1].set_label(':'.join([r_ns, r_relation]))

    return rst_tree


def convert2labelled_attachment_tree(rst_tree):
    if len(rst_tree) == 1:
        return rst_tree[0]

    left_rst_tree = rst_tree[0]
    right_rst_tree = rst_tree[1]
    l_ns, l_relation = left_rst_tree.label().split(':')
    r_ns, r_relation = right_rst_tree.label().split(':')
    ns = '-'.join([l_ns, r_ns])
    relation = l_relation if l_relation.lower() != 'span' else r_relation
    label = ':'.join([ns, relation])

    return Tree(label, [convert2labelled_attachment_tree(rst_tree[0]),
                        convert2labelled_attachment_tree(rst_tree[1])])


def preprocess_rst_dis_format(rst_tree_str):
    rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
    rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
    tree = ParentedTree.fromstring(rst_tree_str, leaf_pattern=r"\_!.+?\_!|[^ ()]+")
    edus, tree = reformat_rst_tree(tree)
    assert all(e[:2] == '_!' and e[-2:] == '_!' for e in edus)
    edus = [e[2:-2] for e in edus]

    tree = binarize(tree)
    # tree = re_categorize(tree)
    tree = convert2labelled_attachment_tree(tree)
    assert len(edus) == len(tree.leaves())
    return edus, tree._pformat_flat("", "()", False)


def preprocess_convert_rst_tree(rst_str):
    tree = Tree.fromstring(rst_str, remove_empty_top_bracketing=True)
    tree = binarize(tree)
    tree = re_categorize(tree)
    tree = convert2labelled_attachment_tree(tree)
    return tree._pformat_flat("", "()", False)


@click.command()
@click.argument('input-file', default='-', type=click.File('r'))
def main(input_file):
    # initialize the loggers
    logging.basicConfig(level='DEBUG')
    # process the given input file
    rst_tree_str = input_file.read().strip()
    rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
    rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
    tree = ParentedTree.fromstring(rst_tree_str)
    reformat_rst_tree(tree)
    tree = binarize(tree)
    tree = re_categorize(tree)
    tree = convert2labelled_attachment_tree(tree)
    tree._pformat_flat("", "()", False)


if __name__ == '__main__':
    main()
