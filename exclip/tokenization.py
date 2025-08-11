from clip.simple_tokenizer import SimpleTokenizer as ClipSimpleTokenizer
import open_clip
from typing import Union, List
import torch


class OpenAITokenizer(ClipSimpleTokenizer):

    def get_token_ids(self, text: str, append_sot_eot: bool = False):
        token_ids = self.encode(text)
        if append_sot_eot:
            sot_token = self.encoder["<|startoftext|>"]
            eot_token = self.encoder["<|endoftext|>"]
            token_ids = [sot_token] + token_ids + [eot_token]
        return token_ids

    def get_tokens(self, text: str):
        ids = self.get_token_ids(text)
        tokens = [self.decoder[id].replace('</w>', '') for id in ids]
        return tokens
    
    def print_tokens(self, text: str, count_cls: bool = True):
        tokens = self.get_tokens(text)
        s = ''
        for i, t in enumerate(tokens):
            if count_cls:
                i += 1
            s += f'{i}-{t} '
        return s.strip()
    
    def tokenize(
        self, 
        texts: Union[str, List[str]], 
        context_length: int = 77, 
        truncate: bool = False
    ):
        '''copied from clip.tokenize() to have everything in one class'''
       
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class OpenClipTokenizer:
    # TODO: subclass open clip SimpleTokenizer

    def __init__(self, model_name: str):
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def tokenize(self, *args, **kwargs):
        return self(*args, **kwargs)

    def get_token_ids(self, text: str, append_sot_eot: bool = False):
        token_ids = self.tokenizer.encode(text)
        if append_sot_eot:
            token_ids = [self.tokenizer.sot_token_id] + token_ids + [self.tokenizer.eot_token_id]
        return token_ids
    
    def get_tokens(self, text: str):
        ids = self.get_token_ids(text)
        tokens = [self.tokenizer.decoder[id].replace('</w>', '') for id in ids]
        return tokens
    
    def print_tokens(self, text: str):
        tokens = self.get_tokens(text)
        s = ''
        for i, t in enumerate(tokens):
            s += f'{i+1}-{t} '
        return s.strip()