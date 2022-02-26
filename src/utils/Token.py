#!/usr/bin/env python
# coding: utf-8
from .ModelType import ModelType, berts, gpts
from typing import List


def bracket(s: str):
    return f'[[{s}]]'


class Token(object):
    """
    A token is the basic unit being represented at each layer of  
    language models.

    In particular, a document (often a sentence), is made up of 
    types, and one particular type in one particular document is a token,
    which the model represents at every layer of the model.
    """

    def __init__(self, doc: List[str], pos: int, model: ModelType):
        self.doc = doc
        self.pos = pos
        self.model = model
        self.type = doc[pos]
        self.identity = doc[pos]
        self.word = doc[pos]
        self.len = len(self.doc)
        self.next = doc[pos + 1] if pos + 1 <= len(doc) - 1 else None
        self.prev = doc[pos - 1] if pos - 1 >= 0 else None
        self.text = self.get_text()
        self.short_text = self.get_short_text()
        self.is_null = Token.is_type_null(self.type)
        self.has_null = any([Token.is_type_null(type_) for type_ in doc])
        self.is_special = Token.is_type_special(self.type)
        self.is_partial = ((model in berts and self.type.startswith('##'))
                           or model in gpts and not self.type.startswith(' ') and pos != 0)
        self.is_edge = pos == 0 or pos == self.len - 1
        self.is_valid = not self.has_null and not self.is_edge and not self.is_special

    def replace(self, type_, pos=None):
        '''Replace this type with the indicated type, not in place.'''
        pos = pos if pos else self.pos
        new_doc = self.doc.copy()
        new_doc[pos] = type_
        return Token(new_doc, self.pos, self.model)

    def get_text(self):
        doc = self.doc.copy()
        doc[self.pos] = bracket(self.type)
        if self.model in gpts:
            return ''.join(doc)
        elif self.model in berts:
            return ''.join([type_.replace('##', '/') if type_.startswith('##') else f' {type_}' for type_ in doc])
        else:
            raise ValueError('Not a berts or a gpt')

    def get_short_text(self, n_context_tokens=2):
        """Get an abbreviated string representation of the context:
        the part of the doc around the embedded token, with emphasis on the embedded token."""
        start_index = self.pos - n_context_tokens
        if start_index >= 0:
            # we have a complete abbreviated context
            new_tok_pos = n_context_tokens
        else:
            # we do not have a complete abbreviated context;
            # abbreviated context will start at beginning of tokens
            start_index = 0
            new_tok_pos = self.pos
        end_index = min(self.pos + n_context_tokens + 1, self.len)
        short_doc = self.doc.copy()[start_index: end_index]
        short_doc[new_tok_pos] = bracket(self.type)
        if self.model in gpts:
            short_text = ''.join(self.doc)
        elif self.model in berts:
            short_text = ' '.join(self.doc)
        return '...' + short_text + '...'

    def get_tok_at_pos(self, pos):
        return Token(self.doc, pos, self.model)

    @staticmethod
    def is_type_special(type_):
        return type_ in ['[CLS]', '[SEP]']

    @staticmethod
    def is_type_null(type_):
        return type_ in ['\n', ]

    @staticmethod
    def is_type_partial(type_, model_type):
        # This method implementationhas been commented out because you cannot determine partial purely from word.
        # This is because for GPT both partial words and sometimes first words (depending on tokenization)
        # both begin without a space - which as far as I know is the only indicator of partialness.
        #
        # Original: return ((model_type in berts and type_.startswith('##')) or
        #             (model_type in gpts and not type_.startswith(' ')))
        pass

    def in_contexts(self, toks):
        return [tok.replace(self.type) for tok in toks]

    def with_types(self, types):
        return [self.replace(type_) for type_ in types]

    def same_next(self, tok):
        return self.next == tok.next

    def same_prev(self, tok):
        return self.prev == tok.prev

    def same_type(self, tok):
        return self.type == tok.type

    def __str__(self):
      return self.get_short_text()
