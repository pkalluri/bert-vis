"""Helpers for handling contexts. Contexts are typically tuples (doc's tokens, index of embedded token)"""
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from utils import html_util


def bracket(s):
    return f'[[{s}]]'


def doc_str(toks):
    return contexts_str(toks, [])


def context_str(toks, pos, marker=bracket, masker=None, masker_marker=None, token_styler=lambda a: a):
    """Get a string representation of the context: i.e. the doc with emphasis on the embedded token."""
    return contexts_str(toks, [pos], marker=marker, masker=masker, masker_marker=masker_marker, token_styler=token_styler)


def contexts_str(toks, positions, marker=bracket, masker=None, masker_marker=None, token_styler=lambda a: a):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    s = ''
    for i, tok in enumerate(toks):
        cleaned_tok:str = tok
        if cleaned_tok.startswith('##'):
            cleaned_tok = f'/{tok[2:]}'
        else:
            cleaned_tok = ' ' + cleaned_tok
        cleaned_tok = token_styler(cleaned_tok)
        if masker and i not in positions and tok == '[MASK]':
            cleaned_tok = masker(cleaned_tok)
        if i in positions:
            if masker_marker and tok == '[MASK]':
                cleaned_tok = masker_marker(cleaned_tok)
            else:
                cleaned_tok = marker(cleaned_tok)
        s += cleaned_tok
    return s


def context_plaintext(toks, pos):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return context_str(toks, pos)


def context_html(toks, pos, marker=html_util.highlighter(), masker=html_util.highlighter('black'), masker_marker=None, token_styler=lambda a: a):
    """Get a html representation of the context: the doc with emphasis on the embedded token."""
    return context_str(toks, pos, marker=marker, masker=masker, masker_marker=masker_marker, token_styler=token_styler)


def abbreviated_context_html(toks, pos, n_context_tokens=2, marker=html_util.highlighter(), masker=None, masker_marker=None, token_styler=lambda a: a):
    """Get an abbreviated html representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    return abbreviated_context(toks, pos, n_context_tokens=n_context_tokens, marker=marker, masker=masker, masker_marker=masker_marker, token_styler=token_styler)

def abbreviated_context(toks, pos, n_context_tokens=2, marker=bracket, masker=None, masker_marker=None, token_styler=lambda a: a):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    start_index = pos - n_context_tokens
    if start_index >= 0:
        # we have a complete abbreviated context
        new_tok_pos = n_context_tokens
    else:
        # we do not have a complete abbreviated context;
        # abbreviated context will start at beginning of tokens
        start_index = 0
        new_tok_pos = pos
    end_index = min(pos + n_context_tokens + 1, len(toks))
    return context_str(toks[start_index: end_index], new_tok_pos, marker=marker, masker=masker,
                       masker_marker=masker_marker, token_styler=token_styler)


def get_doc(contexts, acts, i):
    """
    Get the ith doc's contexts and activations.
    """
    doc_idx = -1
    for context_idx, (toks, pos) in enumerate(contexts):
        if pos == 0:
            doc_idx += 1
        if doc_idx == i:
            doc = toks
            doc_acts = {layer: acts[layer][context_idx: context_idx+len(doc)] for layer in acts}
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
