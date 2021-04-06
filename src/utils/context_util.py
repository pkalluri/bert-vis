"""Helpers for handling contexts. Contexts are typically tuples (doc's tokens, index of embedded token)"""


def context_str(toks, tok_i):
    """Get a string representation of the context: the doc with emphasis on the embedded token."""
    return ' '.join([tok if i != tok_i else f'[[{tok}]]' for (i, tok) in enumerate(toks)])


def abbreviated_context(toks, tok_i, n_context_tokens=2):
    """Get an abbreviated string representation of the context:
    the part of the doc around the embedded token, with emphasis on the embedded token."""
    return context_str(toks[tok_i - n_context_tokens: tok_i + n_context_tokens + 1], n_context_tokens)
