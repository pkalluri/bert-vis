#!/usr/bin/env python
# coding: utf-8
from .ModelType import ModelType, berts, gpts
from typing import List


def bracket(s:str):
    return f'[[{s}]]'


class Token(object):
    """
    A token is the basic unit being represented at each layer of  
    language models.

    In particular, a document (often a sentence), is made up of 
    types, and one particular type in one particular document is a token,
    which the model represents at every layer of the model.
    """
    def __init__(self, doc:List[str], pos:int, model:ModelType):
        self.doc = doc
        self.pos = pos
        self.model = model
        self.type = doc[pos]
        self.identity = doc[pos]
        self.len = len(self.doc)
        self.next = doc[pos+1] if pos+1 <= len(doc)-1 else None
        self.prev = doc[pos-1] if pos-1 >= 0 else None
        self.text = self.get_text()
        self.short_text = self.get_short_text()
        self.is_special = Token.is_type_special(self.type, self.model)
        self.is_partial = ((model in berts and self.type.startswith('##'))
                           or model in gpts and not self.type.startswith(' ') and pos != 0)
        self.is_edge = pos==0 or pos==self.len-1
    

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
            return ''.join([type_.replace('##','/') if type_.startswith('##') else f' {type_}' for type_ in doc])
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
    def is_type_special(type_, model_type):
        return type_ in ['[CLS]', '[SEP]', '\n']
    
#     @staticmethod
#     def is_type_partial(type_, model_type):
#         return ((model in berts and type_.startswith('##')) or
#                 (model in gpts and type_.startswith))
#         elif model in gpts:
#             self.is_partial = not self.type.startswith(' ') and pos != 0
#         return (model_type in berts and type_ in ['[CLS]', '[SEP]'])
    
    
    def in_contexts(self, toks):
        return [tok.replace(self.type) for tok in toks]

    
    def with_types(self, types):
        return [self.replace(type_) for type_ in types]
    
    
    def same_next(self, tok):
        return self.next == tok.next
    
    
    def same_type(self, tok):
        return self.type == tok.type







    # Representations

    # def str(self, marker=bracket, masker=None, 
    #           masker_marker=None, token_styler=lambda a: a):
    #   """Get a string representation of the tok: 
    #   the doc with emphasis on the embedded token."""
    #   s = ''
    #   for i, tok in enumerate(self.doc):
    #       cleaned_tok:str = tok
    #       if cleaned_tok.startswith('##'):
    #           cleaned_tok = f'/{tok[2:]}'
    #       else:
    #           cleaned_tok = ' ' + cleaned_tok
    #       cleaned_tok = token_styler(cleaned_tok)
    #       if masker and i != self.pos and tok == '[MASK]':
    #           cleaned_tok = masker(cleaned_tok)
    #       if i == self.pos:
    #           if masker_marker and tok == '[MASK]':
    #               cleaned_tok = masker_marker(cleaned_tok)
    #           else:
    #               cleaned_tok = marker(cleaned_tok)
    #       s += cleaned_tok
    #   return s


    # def plaintext(self):
    #   """Get a string representation of the tok: 
    #   the doc with emphasis on the embedded token."""
    #   return str(self)


    # def html(self, marker=html_util.highlighter(), 
    #           masker=html_util.highlighter('black'), masker_marker=None, 
    #           token_styler=lambda a: a):
    #   """Get a html representation of the tok: 
    #   the doc with emphasis on the embedded token."""
    #   return str(self, marker=marker, masker=masker, 
    #       masker_marker=masker_marker, token_styler=token_styler)


    # def abbreviated_html(self, n_tokens=2, 
    #                       marker=html_util.highlighter(), 
    #                       masker=None, masker_marker=None, 
    #                       token_styler=lambda a: a):
    #   """Get an abbreviated html representation of the tok:
    #   the part of the doc around the embedded token, 
    #   with emphasis on the embedded token."""
    #   return abbreviated_tok(self, n_tokens=n_tokens, marker=marker, 
    #       masker=masker, masker_marker=masker_marker, token_styler=token_styler)

    # def abbreviated_tok(self, n_tokens=2, marker=bracket, masker=None, 
    #                   masker_marker=None, token_styler=lambda a: a):
    #   """Get an abbreviated string representation of the tok:
    #   the part of the doc around the embedded token, with 
    #   emphasis on the embedded token."""
    #   start_index = self.pos - n_tokens
    #   if start_index >= 0:
    #       # we have a complete abbreviated tok
    #       new_pos = n_tokens
    #   else:
    #       # we do not have a complete abbreviated tok;
    #       # abbreviated tok will start at beginning of tokens
    #       start_index = 0
    #       new_pos = self.pos
    #   end_index = min(self.pos + n_tokens + 1, len(self.doc))
    #   return '...'+str(self.doc[start_index: end_index], new_pos, marker=marker, 
    #                       masker=masker,
    #                       masker_marker=masker_marker, 
    #                       token_styler=token_styler)+'...'

    # def __str__(self):
    #   return self.str()


    # # Comparisons

    # def same_predecessor(self, tok2):
    #   return self.doc[self.pos-1]  == tok2.doc[tok2.pos-1]


    # def same_type(self, tok2):
    #   return self.type == tok2.type


    # def same_successor(self, tok2):
    #   return (len(self.doc)>self.pos+1 and 
    #           self.doc[self.pos+1]  == tok2.doc[tok2.pos+1])


    # # Boolean attributes

    # def is_color(self):
    #   return (tok.type in 
    #           ['red', 'orange', 'yellow', 'green', 'blue', 
    #           'purple', 'black', 'brown'])


