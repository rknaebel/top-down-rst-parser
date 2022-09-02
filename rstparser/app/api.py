import argparse
import io
from typing import List

import click
import nltk
import spacy
import uvicorn
from conllu import TokenList
from conllu.models import Token, Metadata
from conllu.parser import DEFAULT_FIELDS
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nltk.treeprettyprinter import TreePrettyPrinter
from pydantic.main import BaseModel

from rstparser.dataset.merge_file import Doc
from rstparser.networks.hierarchical import HierarchicalParser


def get_argparser():
    parser = argparse.ArgumentParser(description="span based rst parser")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--hierarchical-type', choices=['d2e', 'd2s2e', 'd2p2s2e'], required=True)
    parser.add_argument('--use-hard-boundary', action='store_true')
    parser.add_argument('--model-path', required=True, nargs='+')
    return parser


def get_sentences(edus):
    sentences = []
    edus_tmp = []
    for edu in edus:
        edus_tmp.append(edu.replace('<P>', ''))
        if any(edu.endswith(e) for e in [".", "!", "?", "<P>"]):
            sentences.append(' '.join(edus_tmp))
            edus_tmp = []
    if edus_tmp:
        sentences.append(' '.join(edus_tmp))
    return sentences


def merge_edus_into_parses(edus: List[str], parses):
    result = []
    edu_i = 0
    par_i = 1
    doc_id = str(hash(' '.join(edus)))
    meta = {
        'newdoc id': doc_id,
        'newpar id': f'{doc_id}-p{par_i}'
    }
    for sent_i, sent in enumerate(parses):
        edu = edus[edu_i].replace('<P>', '')
        edu_length = len(edu) + 1
        tokens = []
        for tok_i, tok in enumerate(sent):
            misc = {}
            if tok_i == 0:
                misc['BeginSeg'] = 'YES'
            if tok.idx + len(tok) > edu_length:
                misc['BeginSeg'] = 'YES'
                edu_i += 1
                edu = edus[edu_i].replace('<P>', '')
                if edus[edu_i - 1].endswith('<P>'):
                    par_i += 1
                    meta['newpar id'] = f'{doc_id}-p{par_i}'
                edu_length += len(edu) + 1
            tokens.append(Token(
                id=tok_i + 1,
                form=tok.text,
                lemma="_",
                upos="_",
                xpos="_",
                feats='_',
                head="_",
                deprel="_",
                deps='_',
                misc=misc))
        meta['sent_id'] = str(sent_i)
        meta['text'] = sent.text
        result.append(TokenList(tokens, metadata=Metadata(meta), default_fields=DEFAULT_FIELDS))
        meta = {}
        edu_i += 1
    return result


def merge_as_text(parses: List[TokenList]):
    return ''.join(sent.serialize() for sent in parses)


app = FastAPI(
    title="rst-todo-service",
    version="1.0.0",
    description="RST-based top-down discourse parser.",
)
templates = Jinja2Templates(directory="rstparser/app/pages")

rst_parser: HierarchicalParser = None
nlp: spacy.pipeline.Pipe = None


@app.on_event("startup")
async def startup_event():
    global rst_parser, brown_clusters, nlp
    args, _ = get_argparser().parse_known_args()
    rst_parser = HierarchicalParser.load_model(args.model_path, args)
    nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser", "lemmatizer", "ner"])


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ParserRequest(BaseModel):
    edus: List[str]


def _replace_edu_strings(input_tree: nltk.Tree, edus):
    """Replace EDU strings (i.e., the leaves) with indices."""
    tree = input_tree.copy(deep=True)
    for subtree in tree.subtrees():
        if isinstance(subtree[0], str):
            edu = edus[int(subtree.pop())]
            subtree.append(edu)
    return tree


@app.post("/api/parse")
async def get_parsing(r: ParserRequest):
    """Description
    """
    edus = [edu.strip() for edu in r.edus]
    sentences = get_sentences(edus)
    parses = [nlp(sent) for sent in sentences]
    parses = merge_as_text(merge_edus_into_parses(edus, parses))
    doc = Doc.from_conll_file(io.StringIO(parses))
    pred_rst = rst_parser.parse(doc)
    return {
        "string": pred_rst.pformat(margin=128),
        "pretty": TreePrettyPrinter(pred_rst).text(),
        "text": _replace_edu_strings(pred_rst, edus).pformat(margin=128)

    }


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("hostname")
@click.option("--port", default=8080, type=int)
@click.option("--debug", is_flag=True)
def main(hostname, port, debug):
    uvicorn.run("rstparser.app.api:app", host=hostname, port=port, log_level="debug" if debug else "info", reload=debug)


if __name__ == '__main__':
    main()
