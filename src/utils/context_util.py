"""Helpers for handling contexts. Contexts are typically tuples (doc's tokens, index of embedded token)"""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from utils import html_util


def bracket(s):
    return f'[[{s}]]'


def doc_str(toks):
    return context_str(toks, -1)


def context_str(toks, tok_pos, marker=bracket):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    s = ''
    for i, tok in enumerate(toks):
        cleaned_tok: str = tok
        if cleaned_tok.startswith('##'):
            cleaned_tok = f'/{tok[2:]}'
            if i == tok_pos: cleaned_tok = marker(cleaned_tok)
        else:
            if i == tok_pos: cleaned_tok = marker(cleaned_tok)
            cleaned_tok = ' ' + cleaned_tok
        s += cleaned_tok
    return s


def multi_context_str(doc, positions, marker=bracket):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    s = ''
    for i, tok in enumerate(doc):
        cleaned_tok: str = tok
        if cleaned_tok.startswith('##'):
            cleaned_tok = f'/{tok[2:]}'
            if i in positions: cleaned_tok = marker(cleaned_tok)
        else:
            if i in positions: cleaned_tok = marker(cleaned_tok)
            cleaned_tok = ' ' + cleaned_tok
        s += cleaned_tok
    return s


def context_plaintext(toks, tok_pos):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return context_str(toks, tok_pos, bracket)


def context_html(doc, pos, highlighter=html_util.highlight):
    """Get a html representation of the context: the doc with emphasis on the embedded token."""
    return context_str(doc, pos, highlighter)


def abbreviated_context_html(toks, tok_pos, n_context_tokens=2):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    start_index = tok_pos - n_context_tokens
    if start_index >= 0:
        # we have a complete abbreviated context
        new_tok_pos = n_context_tokens
    else:
        # we do not have a complete abbreviated context;
        # abbreviated context will start at context's first token
        start_index = 0
        new_tok_pos = tok_pos
    end_index = min(tok_pos + n_context_tokens + 1, len(toks))
    return context_html(toks[start_index: end_index], new_tok_pos)


def abbreviated_context(toks, tok_pos, n_context_tokens=2):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    start_index = tok_pos - n_context_tokens
    if start_index >= 0:
        # we have a complete abbreviated context
        new_tok_pos = n_context_tokens
    else:
        # we do not have a complete abbreviated context;
        # abbreviated context will start at beginning of tokens
        start_index = 0
        new_tok_pos = tok_pos
    end_index = min(tok_pos + n_context_tokens + 1, len(toks))
    return context_str(toks[start_index: end_index], new_tok_pos)


def get_doc(contexts, acts, i, layers=None):
    """
    Get the ith doc's contexts and activations.
    """
    if not layers:
        layers = acts.keys()
    doc_idx = -1
    for context_idx, (toks, pos) in enumerate(contexts):
        if pos == 0:
            doc_idx += 1
        if doc_idx == i:
            doc = toks
            doc_acts = {layer: acts[layer][context_idx: context_idx+len(doc)] for layer in layers}
            break
    return doc, doc_acts


def get_doc_ids(contexts, i):
    """
    Get the ith doc's contexts and activations.
    """
    doc_number = -1
    for context_idx, (toks, pos) in enumerate(contexts):
        if pos == 0:
            doc_number += 1
        if doc_number == i:
            return list(range(context_idx, context_idx+len(toks)))
