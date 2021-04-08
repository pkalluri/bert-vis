"""Helpers for handling contexts. Contexts are typically tuples (doc's tokens, index of embedded token)"""


def context_str(toks, tok_i):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return ' '.join([tok if i != tok_i else f'[[{tok}]]' for (i, tok) in enumerate(toks)])


def emphasized_html(s):
    return f'<b style="color: hotpink">{s}</b>'


def context_html(toks, tok_i):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return ' '.join([tok if i != tok_i else f' {emphasized_html(tok)} ' for (i, tok) in enumerate(toks)])


def abbreviated_context_html(toks, tok_i, n_context_tokens=2):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    start_index = max(tok_i - n_context_tokens, 0)
    end_index = min(tok_i + n_context_tokens + 1, len(toks))
    return context_html(toks[start_index: end_index], n_context_tokens)


def abbreviated_context(toks, tok_i, n_context_tokens=2):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    start_index = max(tok_i - n_context_tokens, 0)
    end_index = min(tok_i + n_context_tokens + 1, len(toks))
    return context_str(toks[start_index: end_index], n_context_tokens)
