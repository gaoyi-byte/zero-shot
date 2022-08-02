import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim, tensor
#from . import network
from transformers import BertTokenizer, BertModel,BertForMaskedLM


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length,train_type): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.mlm = BertForMaskedLM.from_pretrained(pretrain_path)
        self.train_type=train_type
        #self.m = nn.LogSoftmax(dim=1)
        #self.cost = nn.NLLLoss()

    def forward(self, inputs):
        
        #x = self.mlm(inputs['word'], attention_mask=inputs['mask'],token_type_ids=inputs['seg'],output_hidden_states=True).hidden_states[-1] 
        x = self.bert(inputs['word'], attention_mask=inputs['mask'],token_type_ids=inputs['seg'])[0]
        return x
    
    def id_forward(self, inputs):
        
        #x = self.mlm(inputs,output_hidden_states=True).hidden_states[-1] 
        x = self.bert(inputs)[0]
        return x
    
    def tem_tokenize(self, text,cal_text,max_length=None):
        # token -> index
        if max_length is None:
            max_length=self.max_length
        text=text.split('<span>')
        tokens = ['[CLS]']
        tokens+=self.tokenizer.tokenize(text[0])
        begin=len(tokens)
        tokens+=self.tokenizer.tokenize(cal_text)
        end=len(tokens)
        tokens+=self.tokenizer.tokenize(text[1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        indexed_tokens=torch.tensor(indexed_tokens)
        return indexed_tokens, begin,end

    def qa_prompt_tokenize(self,cal_text,text,mask_len):
        # token -> index
        max_length=self.max_length
        cal_text=cal_text+'是'+'[MASK]'*mask_len+'。'
        tokens = ['[CLS]']
        tokens+=self.tokenizer.tokenize(cal_text)
        pos=len(tokens)
        tokens+=['[SEP]']
        for token in text:
            tokens+=self.tokenizer.tokenize(token)
        tokens+=['[SEP]']
        
        #padding
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
            #mask.append(0)
        indexed_tokens = indexed_tokens[:max_length]

        # mask
        mask = np.zeros((max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        seg = np.ones((max_length), dtype=np.int32)
        seg[:min(max_length, pos)] = 0
        
        indexed_tokens = torch.tensor(indexed_tokens).long()
        mask = torch.tensor(mask).long()
        seg = torch.tensor(seg).long()
        return indexed_tokens, mask,seg

    def tokenize(self, raw_tokens,max_length=None,prompt=None):
        # token -> index
        if max_length is None:
            max_length=self.max_length
        
        tokens = ['[CLS]']
        if prompt:
            tokens+=self.tokenizer.tokenize(prompt)
            tokens+=['[SEP]']
        pos=len(tokens)
        for token in raw_tokens:
            tokens += self.tokenizer.tokenize(token)
        
        tokens+=['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        #print(tokens,indexed_tokens)

        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
            #mask.append(0)
        indexed_tokens = indexed_tokens[:max_length]

        # mask
        mask = np.zeros((max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        seg = np.ones((max_length), dtype=np.int32)
        if prompt:
            seg[:min(max_length, pos)] = 0
        else:
            seg[:min(max_length, len(raw_tokens) + 1)] = 0
        indexed_tokens = torch.tensor(indexed_tokens).long()
        mask = torch.tensor(mask).long()
        seg = torch.tensor(seg).long()
        return indexed_tokens, mask,seg
        
    