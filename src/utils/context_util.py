"""Helpers for handling contexts. Contexts are typically tuples (doc's tokens, index of embedded token)"""
from src.utils import html_util


def bracket(s):
    return f'[[{s}]]'


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


def context_plaintext(toks, tok_pos):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return context_str(toks, tok_pos, bracket)


def context_html(toks, tok_pos):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return context_str(toks, tok_pos, html_util.highlight_html)


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
