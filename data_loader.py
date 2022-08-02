import logging
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import itertools


class event(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self,types, encoder,root,K,event_types):
        self.root = root
        if types=='train':
            path = os.path.join(root, "train.json")
        elif types=='val':
            path = os.path.join(root, "val.json")
        else:
            path = os.path.join(root, "test.json")
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path, encoding='utf8'))
        self.data=[]
        if types=='train':
            for cls in event_types:
                self.data+=self.json_data[cls][:K]
        else:
            for cls in event_types:
                self.data+=self.json_data[cls]
        self.encoder = encoder

    def __additem__(self, d, word, mask,seg):
        d['word'].append(word)
        d['mask'].append(mask)
        d['seg'].append(seg)
        
    def __getitem__(self, index):
        data = {'word': [], 'mask': [], 'seg': []}
        args_data={'gold_args':[],'cal_args':[]}
        item=self.data[index]
        word, mask,seg = self.encoder.tokenize(item['tokens'])
        self.__additem__(data, word,  mask, seg)
        args_data['gold_args']=item['args']
        args_data['cal_args']=item['cal_args']
        args_data['label']=item['type']
        args_data['text']=item['text']

        return data,args_data
    
    def __len__(self): 
        return len(self.data)

class promt_event(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self,types, encoder,root,K,event_types,schema):
        self.root = root
        if types=='train':
            path = os.path.join(root, "event3-train.json")
        elif types=='val':
            path = os.path.join(root, "event3-val.json")
        else:
            path = os.path.join(root, "event3-test.json")
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path, encoding='utf8'))
        self.data=[]
        if types=='train':
            for cls in event_types:
                self.data+=self.json_data[cls][:K]
        else:
            for cls in event_types:
                self.data+=self.json_data[cls]
        self.encoder = encoder
        self.schema=schema

    def __additem__(self, d, word, mask,seg):
        d['word'].append(word)
        d['mask'].append(mask)
        d['seg'].append(seg)
        
    def __getitem__(self, index):
        data = {'word': [], 'mask': [], 'seg': []}
        args_data={'gold_args':[],'cal_args':[]}
        item=self.data[index]
        mask_len=self.schema[item['type']]['maxl']
        for cal in item['cal_args']:#暂时用gold
            word, mask,seg = self.encoder.qa_prompt_tokenize(cal['text'],item['tokens'],mask_len)
            self.__additem__(data, word,  mask, seg)
        args_data['gold_args']=item['args']
        args_data['cal_args']=item['cal_args']
        args_data['label']=item['type']
        args_data['text']=item['text']

        return data,args_data
    
    def __len__(self): 
        return len(self.data)

def collate_fn_prompt(data):
    batch_data= {'word': [],  'mask': [], 'seg':[]}
    batch_args={'text':[],'label':[],'offset':[],'span_type':[],'num':[]}
    tmp_data, tmp_arg = zip(*data)
    for i in range(len(tmp_data)):
        for k in tmp_data[i]:
            batch_data[k] += tmp_data[i][k]
        for k in tmp_arg[i]:
            batch_args[k].append(tmp_arg[i][k])
        
    for k in batch_data:
        batch_data[k] = torch.stack(batch_data[k], 0)
    return batch_data, batch_args

def collate_fn(data):
    batch_data= {'word': [],  'mask': [], 'seg':[]}
    tmp_data, tmp_arg = zip(*data)
    for i in range(len(tmp_data)):
        #print(len(tmp_data[i]['word']))
        for k in tmp_data[i]:
            batch_data[k] += tmp_data[i][k]
        
    for k in batch_data:
        batch_data[k] = torch.stack(batch_data[k], 0)
    return batch_data, tmp_arg

def get_loader(types,train_type,encoder,batch_size, K,event_types,schema,num_workers=1,collate_fn=collate_fn,root='./data'):
    
    if train_type=='prompt':
        dataset = promt_event(types,encoder,root,K,event_types,schema)
        #collate_fn=collate_fn_prompt
    else:
        dataset = event(types,encoder,root,K,event_types)
    if types=='train':
        flag=True
    else:
        flag=False
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=flag,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader


class event_new(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self,types, encoder,root,K,event_types,offset=10,):
        self.root = root
        if types=='train':
            path = os.path.join(root, "class_train.json")
        elif types=='val':
            path = os.path.join(root, "class_val.json")
        else:
            path = os.path.join(root, "class_val.json")
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path, encoding='utf8'))
        self.data=[]
        if types=='train':
            for cls in event_types:
                self.data+=self.json_data[cls][:K]
        else:
            for cls in event_types:
                self.data+=self.json_data[cls]
        self.encoder = encoder

    def __additem__(self, d, word, mask,seg):
        d['word'].append(word)
        d['mask'].append(mask)
        d['seg'].append(seg)
        
    def __getitem__(self, index):
        data = {'word': [], 'mask': [], 'seg': [],'offset':[],'cal_num':[]}
        args_data={'gold_args':[],'cal_args':[]}
        item=self.data[index]
        word, mask,seg = self.encoder.tokenize(item['tokens'])
        self.__additem__(data, word,  mask, seg)
        args_data['gold_args']=item['args']
        args_data['cal_args']=item['args']
        data['cal_num'].append(len(item['args']))
        for cal in item['args']:
            
            data['offset'].append()



        args_data['label']=item['type']
        args_data['text']=item['text']
        # todo:
        #正则表达式抽取span

        return data,args_data
    
    def __len__(self): 
        return len(self.data)