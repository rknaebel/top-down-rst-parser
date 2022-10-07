import glob
import os
import re
import sys
from pathlib import Path

import click
import stanza
from conllu.models import Token, TokenList, Metadata
from conllu.parser import DEFAULT_FIELDS
from tqdm import tqdm

RE_PAR = re.compile(r"\n\s*(?:\n|\s\s\s|\t)\s*")
RE_CR = re.compile(r"\r+")
RE_WS_PLUS = re.compile(r"\s+")


def split_to_paragraphs(content):
    res = RE_CR.sub(' ', content)
    res = RE_PAR.split(res)
    res = filter(lambda x: x != '.START', map(lambda x: RE_WS_PLUS.sub(' ', x).strip(), res))
    return list(res)


def load_parser():
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    stanza.download(lang='en')
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,constituency')
    sys.stdout = tmp_stdout
    return parser


def convert_parses(fname, parses):
    result = []
    doc_id = str(hash(fname))
    meta = {
        'newdoc id': doc_id,
    }
    sent_off = 0
    for par_i, par in enumerate(parses):
        meta['newpar id'] = f'{doc_id}-p{par_i}'
        for sent_i, sent in enumerate(par.sentences):
            meta['sent_id'] = str(sent_off + sent_i)
            meta['text'] = sent.text
            meta['parse'] = str(sent.constituency)
            tokens = []
            for tok_i, tok in enumerate(sent.words):
                misc = {}
                tokens.append(Token(
                    id=tok_i + 1,
                    form=tok.text,
                    lemma=tok.lemma,
                    upos=tok.upos,
                    xpos=tok.xpos,
                    feats=tok.feats if tok.feats else "_",
                    head=tok.head,
                    deprel=tok.deprel,
                    deps='_',
                    misc=misc))

            result.append(TokenList(tokens, metadata=Metadata(meta), default_fields=DEFAULT_FIELDS))
            meta = {}
        sent_off += len(par.sentences)
    return result


def load_content(fname):
    try:
        content = open(fname).read()
    except UnicodeDecodeError:
        content = open(fname, encoding='latin-1').read()
    if content.startswith('.START'):
        content = content[len('.START'):]
    return content.strip()


@click.command()
@click.argument("source_path", type=str)
@click.argument("target_path", type=click.Path(file_okay=False))
@click.option('-r', '--replace-exist', is_flag=True)
@click.option('-d', '--delete-target', is_flag=True)
@click.option('--path-prefix', type=str, default="")
def main(source_path, target_path, replace_exist, delete_target, path_prefix):
    if delete_target:
        for fn in os.listdir(target_path):
            os.remove(os.path.join(target_path, fn))
    os.makedirs(target_path, exist_ok=True)
    parser = load_parser()
    for fname in tqdm(glob.glob(source_path)):
        paragraphs = split_to_paragraphs(load_content(fname))
        parses = []
        for content in paragraphs:
            parses.append(parser(content))

        if path_prefix:
            fname = fname[len(path_prefix):].replace(os.path.sep, '-').strip('-')
        else:
            fname = os.path.basename(fname)

        parses = convert_parses(fname, parses)
        dest = Path(os.path.join(target_path, fname)).with_suffix('.conll')
        with dest.open('w') as fh:
            for sent in parses:
                fh.write(sent.serialize())


if __name__ == '__main__':
    main()
