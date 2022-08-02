from cmath import sin
import sys
sys.path.append('..')
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import json

def similarity(x, y, dim1,dis):
    if dis=='ou':
        return -(torch.pow(x - y, 2)).sum(dim1)#欧式距离
    elif dis=='dot':
        return (x * y).sum(dim1)
    elif dis=='cos':
        return torch.cosine_similarity(x, y,dim=dim1)
    else:
        return NotImplementedError

def span2type(cal_text,text,encoder,type_emb,span_types,prompt_tem,dis='cos'):
    if cal_text in '股票' or cal_text in '股份':
        return 'V-股票'
    elif bool(re.search('[年月日]',cal_text)):
        return 'T-DATE' 
    elif '%' in cal_text:
        return 'T-NUM-PERCENT'
    elif bool(re.search('[$￥元]',cal_text)):
        return 'T-NUM-MONEY'
    elif bool(re.search(r'\d',cal_text)):
        return 'T-NUM'
    else:
        #return 'T-O'
        span_emb=[]
        for tem in prompt_tem:
            inputs,begin,end=encoder.tem_tokenize(tem,cal_text)
            inputs=inputs.cuda().unsqueeze(0)
            ver_emb=encoder.id_forward(inputs).squeeze(0)[begin:end].mean(0)
        span_emb.append(ver_emb)
        span_emb=torch.stack(span_emb,0)
        logits=similarity(span_emb.mean(0),type_emb,-1,dis)
        _,pred = torch.max(logits, -1)
        
        return span_types[pred]

def cal_ver_emb(encoder):
    encoder=encoder.cuda()
    types=[json.loads(line) for line in open('data/seen_schema/types.json', encoding='utf8')]
    prompt_tem=[]
    for span_type in types[:3]:
        prompt_tem+=span_type['prompts']
    type_emb=[]
    span_types=[]
    for span_type in types[:3]:
        span_emb=[]
        span_types.append(span_type['type'])
        for ver in span_type['verbalizers']:
            for tem in prompt_tem:
                inputs,begin,end=encoder.tem_tokenize(tem,ver)
                inputs=inputs.cuda().unsqueeze(0)
                ver_emb=encoder.id_forward(inputs).squeeze(0)[begin:end].mean(0)
                span_emb.append(ver_emb)
        span_emb=torch.stack(span_emb,0)
        type_emb.append(span_emb.mean(0))
    type_emb=torch.stack(type_emb,0)
    return type_emb,span_types,prompt_tem

class cls(nn.Module):
    
    def __init__(self, sentence_encoder, event_type,schema_dep,schema_tem,dis='ou',offset=10,dep=False):
        nn.Module.__init__(self)
        self.cost = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.dis = dis
        self.encoder=sentence_encoder
        self.offset=offset
        self.ver_emb=None
        self.schema_dep=schema_dep
        self.schema_tem=schema_tem
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(sentence_encoder)
        self.schema_info=self.update_schema()
        self.classfier=nn.Linear(768,len(schema_tem[event_type]['参数']),bias=False)
        if dep:
            print(self.classfier.weight.shape,self.schema_info[event_type]['emb'].shape)
            #torch.nn.init.constant_(self.classfier.weight, self.schema_info[event_type]['emb'])
            self.classfier.weight=torch.nn.Parameter(self.schema_info[event_type]['emb'])

    
   
    
    def update_schema(self):
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(self.encoder)
        self.schema_info={}
        for event_type in self.schema_tem:
            schema_emb=[]
            type2label={}
            k=0
            for key,value in self.schema_dep['事件'][event_type]['参数'].items():
                inputs=self.encoder.tokenizer(key,key+value+key,return_tensors="pt")
                for key1 in inputs:
                    inputs[key1]=torch.tensor(inputs[key1]).cuda()
                '''
                arg_emb=model.bert(**inputs,output_hidden_states=True).hidden_states
                arg_emb=arg_emb[-1]
                arg_emb=arg_emb.mean(1).squeeze(0)
                '''
                arg_emb=self.encoder.bert(**inputs)[0].mean(1).squeeze(0)
                schema_emb.append(arg_emb)
                type2label[key]=k
                k+=1
            schema_emb=torch.stack(schema_emb,0)
            self.schema_info[event_type]={}
            self.schema_info[event_type]['emb']=schema_emb
            self.schema_info[event_type]['types']=list(self.schema_dep['事件'][event_type]['参数'].keys())
            self.schema_info[event_type]['type2label']=type2label
            
        return self.schema_info



    def forward(self, data,args,mask=False):
        sen_embs=self.encoder(data)
        all_logits=[]
        all_label=[]
        for i,sen_emb in enumerate(sen_embs):
            span=[]
            label=[]
            mask_type=torch.zeros((len(args[i]['cal_args']),len(self.schema_tem[args[i]['label']]['参数']))).cuda()
            for t1,cal in enumerate(args[i]['cal_args']):
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-self.offset):index[-1]+1+self.offset]
                if len(cal_emb)==0:
                    cal_emb=sen_emb[0]
                else:
                    cal_emb=cal_emb.mean(0)
                span.append(cal_emb) 
                if mask:
                    span_type=cal['span_type']
                    #span_type=span2type(cal['text'],args[i]['text'],self.encoder,self.type_emb,self.span_types,self.prompt_tem,self.dis)
                    #cal['span_type']=span_type
                    for t2,(arg_type,arg_text) in enumerate(self.schema_tem[args[i]['label']]['参数'].items()):
                        arg_span_type=arg_text["span_type"]
                        if '^' in arg_span_type[0]:
                            if '^'+span_type in arg_span_type:
                                mask_type[t1][t2]=0
                            else:
                                mask_type[t1][t2]=1
                        elif span_type in arg_span_type or 'MISC' in arg_span_type:
                            mask_type[t1][t2]=1
                label.append(self.schema_info[args[i]['label']]['type2label'][cal['type']])


            span=torch.stack(span,0)#候选,(M,d)
            logits=self.classfier(span)
            if mask:
                logits=logits*mask_type
            all_logits.append(logits)
            all_label.append(torch.tensor(label).cuda())
        return sen_embs,all_logits,all_label

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        loss = self.cost(logits, label.view(-1))
        return loss

    def margin_loss(self,logits,label):
        self.M=-100
        N = logits.shape[-1]
        logits = logits
        labels = F.one_hot(label, N)
        labels = (0.5-labels)*2
        loss = torch.sum(torch.clamp(labels.mul(logits), min=0))/N
        return loss

class dep(nn.Module):
    
    def __init__(self, sentence_encoder, schema_dep,schema_tem,dis='ou',offset=10,weight=False):
        nn.Module.__init__(self)
        self.cost = nn.CrossEntropyLoss()
        self.dis = dis
        self.encoder=sentence_encoder
        self.offset=offset
        self.ver_emb=None
        self.schema_dep=schema_dep
        self.schema_tem=schema_tem
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(sentence_encoder)
        self.schema_info=self.update_schema()
        self.weight=weight
        if weight:
            self.classifier=nn.Linear(768,128)

    
    def update_schema(self):
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(self.encoder)
        self.schema_info={}
        for event_type in self.schema_tem:
            schema_emb=[]
            type2label={}
            k=0
            for key,value in self.schema_dep['事件'][event_type]['参数'].items():
                inputs=self.encoder.tokenizer(key,key+value+key,return_tensors="pt")
                for key1 in inputs:
                    inputs[key1]=torch.tensor(inputs[key1]).cuda()
                arg_emb=self.encoder.bert(**inputs)[0].mean(1).squeeze(0)
                schema_emb.append(arg_emb)
                type2label[key]=k
                k+=1
            schema_emb=torch.stack(schema_emb,0)
            self.schema_info[event_type]={}
            self.schema_info[event_type]['emb']=schema_emb
            self.schema_info[event_type]['types']=list(self.schema_dep['事件'][event_type]['参数'].keys())
            self.schema_info[event_type]['type2label']=type2label
            
        return self.schema_info

    def forward(self, data,args,mask=False):
        sen_embs=self.encoder(data)
        all_logits=[]
        all_label=[]
        for i,sen_emb in enumerate(sen_embs):
            span=[]
            label=[]
            mask_type=torch.zeros((len(args[i]['cal_args']),len(self.schema_tem[args[i]['label']]['参数']))).cuda()
            for t1,cal in enumerate(args[i]['cal_args']):
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-self.offset):index[-1]+1+self.offset]
                if len(cal_emb)==0:
                    cal_emb=sen_emb[0]
                else:
                    cal_emb=cal_emb.mean(0)
                span.append(cal_emb) 
                if mask:
                    span_type=cal['span_type']
                    for t2,(arg_type,arg_text) in enumerate(self.schema_tem[args[i]['label']]['参数'].items()):
                        arg_span_type=arg_text["span_type"]
                        if '^' in arg_span_type[0]:
                            if '^'+span_type in arg_span_type:
                                mask_type[t1][t2]=0
                            else:
                                mask_type[t1][t2]=1
                        elif span_type in arg_span_type or 'MISC' in arg_span_type:
                            mask_type[t1][t2]=1
                #label.append(self.schema_info[args[i]['label']]['type2label'][cal['type']])
                
            if span ==[]:
                logits=[]
                all_logits.append(logits)
                continue
            span=torch.stack(span,0).unsqueeze(1)#候选,(M,d)
            arg_embs=self.schema_info[args[i]['label']]['emb'].unsqueeze(0)
            if self.weight:
                span=self.classifier(span)
                arg_embs=self.classifier(self.schema_info[args[i]['label']]['emb'])

            logits=similarity(arg_embs,span,-1,self.dis)#(B,N,M)
            #logits=similarity(self.schema_info[args[i]['label']]['emb'].unsqueeze(0),span.unsqueeze(1),-1,self.dis)#(B,N,M)
            if mask:
                logits=logits*mask_type
            all_logits.append(logits)
            #all_label.append(torch.tensor(label).cuda())
        return sen_embs,all_logits,all_label

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        loss = self.cost(logits, label.view(-1))
        return loss

    def margin_loss(self,logits,label):
        self.M=-100
        N = logits.shape[-1]
        logits = logits
        labels = F.one_hot(label, N)
        labels = (0.5-labels)*2
        loss = torch.sum(torch.clamp(labels.mul(logits), min=0))/N
        return loss

class tem(nn.Module):
    
    def __init__(self, sentence_encoder, schema_dep,schema_tem,dis='ou',offset=10,weight=False):
        nn.Module.__init__(self)
        self.cost = nn.CrossEntropyLoss()
        self.dis = dis
        self.encoder=sentence_encoder
        self.offset=offset
        self.schema_dep=schema_dep
        self.schema_tem=schema_tem
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(sentence_encoder)
        self.schema_info=self.update_schema()
        self.weight=weight
        if weight:
            self.classifier=nn.Linear(768,128)

    
    def update_schema(self):
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(self.encoder)
        self.schema_info={}
        for event_type in self.schema_tem:
            type2label={}
            k=0
            for key,value in self.schema_dep['事件'][event_type]['参数'].items():
                type2label[key]=k
                k+=1
            
            self.schema_info[event_type]={}
            self.schema_info[event_type]['types']=list(self.schema_dep['事件'][event_type]['参数'].keys())
            self.schema_info[event_type]['type2label']=type2label
            
        return self.schema_info

    def forward(self, data,args,mask=False):
        sen_embs=self.encoder(data)
        all_logits=[]
        all_label=[]
        for i,sen_emb in enumerate(sen_embs):
            span=[]
            arg_embs=[]
            label=[]
            mask_type=torch.zeros((len(args[i]['cal_args']),len(self.schema_tem[args[i]['label']]['参数']))).cuda()
            arg_split=self.schema_tem[args[i]['label']]['tem_num']
            for t1,cal in enumerate(args[i]['cal_args']):
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-self.offset):index[-1]+1+self.offset]
                if len(cal_emb)==0:
                    cal_emb=sen_emb[0]
                else:
                    cal_emb=cal_emb.mean(0)
                span.append(cal_emb) 
                for t2,(arg_type,arg_texts) in enumerate(self.schema_tem[args[i]['label']]['参数'].items()):
                    #arg_text=arg_text.replace('<M>',cal['text'])
                    
                    arg_text=arg_texts["tem"]
                    if mask:
                        arg_span_type=arg_texts["span_type"]
                        span_type=cal['span_type']
                        #span_type=span2type(cal['text'],args[i]['text'],self.encoder,self.type_emb,self.span_types,self.prompt_tem,self.dis)

                        if '^' in arg_span_type:
                            if '^'+span_type in arg_span_type:
                                mask_type[t1][t2]=0
                            else:
                                mask_type[t1][t2]=1
                        elif span_type in arg_span_type or 'MISC' in arg_span_type:
                            mask_type[t1][t2]=1
                    for text in arg_text:
                        inputs,begin,end=self.encoder.tem_tokenize(text,cal['text'])
                        inputs=inputs.cuda().unsqueeze(0)
                        arg_emb=self.encoder.id_forward(inputs).squeeze(0)[max(0,begin-self.offset):end+self.offset].mean(0)
                        arg_embs.append(arg_emb)
                #label.append(self.schema_info[args[i]['label']]['type2label'][cal['type']])

            if span ==[]:
                logits=[]
                all_logits.append(logits)
                continue
            arg_embs=torch.stack(arg_embs,0)#候选,(M,d)
            span=torch.stack(span,0).unsqueeze(1)#候选,(M,d)
            arg_embs=arg_embs.reshape(len(args[i]['cal_args']),-1,768)
            if self.weight:
                span=self.classifier(span)
                arg_embs=self.classifier(arg_embs)
            logits=similarity(arg_embs,span,-1,self.dis)#(B,N,M)

            #logits=similarity(arg_embs,span.unsqueeze(1),-1,self.dis)#(M,N)
            logits=torch.split(logits, arg_split,dim=1)
            f=[]
            for tmp in logits:
                aaa,_=torch.max(tmp,-1)
                f.append(aaa)
            logits=torch.stack(f,1)
            if mask:
                logits=logits*mask_type
            all_logits.append(logits)
            #all_label.append(torch.tensor(label).cuda())
        return sen_embs,all_logits,all_label

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        loss = self.cost(logits, label.view(-1))
        return loss

    def margin_loss(self,logits,label):
        self.M=-100
        N = logits.shape[-1]
        logits = logits
        labels = F.one_hot(label, N)
        labels = (0.5-labels)*2
        loss = torch.sum(torch.clamp(labels.mul(logits), min=0))/N
        return loss

class prompt(nn.Module):
    
    def __init__(self, sentence_encoder, schema_dep,schema_tem,dis='ou',offset=10):
        nn.Module.__init__(self)
        self.cost = nn.CrossEntropyLoss()
        self.dis = dis
        self.encoder=sentence_encoder
        self.offset=offset
        self.schema_dep=schema_dep
        self.schema_tem=schema_tem 
        self.type_emb,self.span_types,self.prompt_tem=cal_ver_emb(sentence_encoder)
        self.schema_info=self.update_schema()

    
    def update_schema(self):
        
        self.schema_info={}#对应的事件要素id
        
        for event_type in self.schema_tem:
            #type_id[i]事件要素对应的token id，type_id[i][j]实验要素i，的第j个token
            self.schema_info[event_type]={}
            all_id=[]
            all_mask=[]
            max_length=self.schema_tem[event_type]['maxl']
            for i,key in enumerate(self.schema_tem[event_type]['参数']):
                key=self.encoder.tokenizer.tokenize(key)
                indexed_tokens=self.encoder.tokenizer.convert_tokens_to_ids(key)
                
                # mask
                mask = torch.zeros(max_length)
                mask[:len(indexed_tokens)] = 1

                while len(indexed_tokens) < max_length:
                    indexed_tokens.append(0)
                    #mask.append(0)
                indexed_tokens = indexed_tokens[:max_length]
                indexed_tokens = torch.tensor(indexed_tokens)
                
                all_id.append(indexed_tokens)
                all_mask.append(mask)
                #print(key,indexed_tokens,mask)

            all_id=torch.stack(all_id,0)
            all_mask=torch.stack(all_mask,0)

            self.schema_info[event_type]['id']=all_id.t()#mask_len,len(arguments)
            self.schema_info[event_type]['mask']=all_mask.t()
            #转置后，type_id[i][j]为第j个要素第i个token对应的词表id

    def forward(self, inputs,args,mask=False):
        #print(inputs)
        logits = self.encoder.mlm(inputs['word'], attention_mask=inputs['mask'],token_type_ids=inputs['seg'],output_hidden_states=True).logits
        print(logits.shape)#(sen,max_length,vocab_size)
        num=0
        res_logits=[]
        for batch, arg in enumerate(args):
            num=num+len(arg['cal_args'])
            mask_token_index =(inputs['word'][num:num+len(arg['cal_args'])] == self.encoder.tokenizer.mask_token_id).nonzero(as_tuple=True)#(行索引，列索引)
            row_index=mask_token_index[0].reshape(len(arg['cal_args']),-1)
            col_index=mask_token_index[1].reshape(len(arg['cal_args']),-1)
            #print(row_index,col_index)
            print('col_index',col_index.shape,len(arg['cal_args']))
            single_logits=logits[row_index,col_index]
            #single_logits=torch.gather(logits[num:num+len(arg['cal_args'])],1,torch.tensor(col_index))#sen,mask_len,vocab_size
            print(single_logits[0].shape)#sen,mask_len,voacb_size
            single_logits=torch.gather(single_logits[0],1,self.schema_info[arg['label']])
            print(single_logits.shape)
            logits=logits.sum(0)


            

        
        
       
        print('mask_id',mask_token_index)

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        loss = self.cost(logits, label.view(-1))
        return loss

    def margin_loss(self,logits,label):
        self.M=-100
        N = logits.shape[-1]
        logits = logits
        labels = F.one_hot(label, N)
        labels = (0.5-labels)*2
        loss = torch.sum(torch.clamp(labels.mul(logits), min=0))/N
        return loss