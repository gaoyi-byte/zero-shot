from cmath import log
from encoder import BERTSentenceEncoder
from data_loader import get_loader
import util
from tqdm import tqdm
import sys
import torch
from torch import logit, optim, nn
import numpy as np
import json
import argparse
import os
import logging
import os
import random
import yaml
from sklearn.metrics import f1_score
import warnings
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')
from torch.nn.utils import clip_grad_norm
import re

def load_definition_schema_file(filename):
    """Load schema file in Yaml
    读取 YAML 定义的 Schema 文件
    """
    return yaml.load(open(filename, encoding='utf8'), Loader=yaml.FullLoader)

def similarity(x, y, dim1,dis):
    if dis=='ou':
        return -(torch.pow(x - y, 2)).sum(dim1)#欧式距离
    elif dis=='dot':
        return (x * y).sum(dim1)
    elif dis=='cos':
        return torch.cosine_similarity(x, y,dim=dim1)
    else:
        return NotImplementedError

def span2type(cal_text,model):
    if bool(re.search('[年月日]',cal_text)):
        return 'T-DATE' 
    elif '%' in cal_text:
        return 'T-NUM-IN-PERCENT'
    elif bool(re.search('[$￥]',cal_text)):
        return 'T-MONEY-NUM'
    elif bool(re.search(r'\d',cal_text)):
        return 'T-NUMBER'
    else:
        inputs = model.tokenizer(f"{cal_text} 是一个 [MASK].", return_tensors="pt")
        for key in inputs:
            inputs[key]=inputs[key].cuda()

        with torch.no_grad():
            logits = model.MLM(**inputs).logits
       
        mask_token_index = (inputs.input_ids == model.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        per_id=model.tokenizer.convert_tokens_to_ids('人')
        org_id=model.tokenizer.convert_tokens_to_ids('机构')
        if logits[0, mask_token_index,per_id]>logits[0, mask_token_index,org_id]:
            return 'T-PER'
        else:
            return 'T-ORG'

def cal_cal_type(opt,event_type,model):
    #比较候选词和候选词
    test_data=json.load(open(f'{data_path}/post_val.json', encoding='utf8'))
    all_f1=[]
    type2label={}
    for index,arg_type in enumerate(schema[event_type]['参数'].keys()):
        type2label[arg_type]=index
    types=list(schema[event_type]['参数'].keys())
    print(type2label)
    for data in tqdm(test_data):
        if data['type']==event_type:
            #print('##########################################################\n')
            #print(f"{event_type}:{data['text']}")
            #print('')
            
            logging.info('##########################################################\n')
            logging.info(f"{event_type}:{data['text']}")
            logging.info('')

            word,mask,seg=model.tokenize(data['tokens'])
            word=word.unsqueeze(0).cuda()
            sen_emb=model.bert(word)[0].squeeze(0)
            arg_embs=[]
            
            label=[]
            span=[]
            mask_type=torch.zeros((len(data['args'])),len(schema[event_type]['参数'])).cuda()
            for t1,cal in enumerate(data['args']):
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                cal_span_type=span2type(cal['text'],model)
                if len(cal_emb)==0:
                    cal_emb=sen_emb
                cal_emb=cal_emb.mean(0)
                span.append(cal_emb)
                for t2,(arg_type,arg_text) in enumerate(schema[event_type]['参数'].items()):
                    #arg_text=arg_text.replace('<M>',cal['text'])
                    arg_span_type=arg_text.split('|')[0]
                    if cal_span_type in arg_span_type or 'MISC' in arg_span_type:
                        mask_type[t1][t2]=1
                    arg_text=arg_text[len(arg_span_type)+1:]
                    inputs,begin,end=model.prompt_tokenize(arg_text,cal['text'])
                    inputs=inputs.cuda().unsqueeze(0)
                    arg_emb=model.bert(inputs)[0].squeeze(0)[begin:end].mean(0)
                    arg_embs.append(arg_emb)
                label.append(type2label[cal['type']])
                
            
            arg_embs=torch.stack(arg_embs,0)#候选,(M,d)
            span=torch.stack(span,0)#候选,(M,d)
               
            arg_embs=arg_embs.reshape(len(data['args']),len(schema[event_type]['参数']),768)
            logits=similarity(arg_embs,span.unsqueeze(1),-1,opt.sim)#(B,N,M)
            logits=logits*mask_type
            _, pred = torch.max(logits, -1) #(N,1)
            label=torch.tensor(label)
            #print(label,pred)
            for i,cal in enumerate( data['args']):
                #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                logging.info((f"{cal['text']} 实体类型：{span2type(cal['text'],model)} | gold标签是：{cal['type']}|预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}"))
                for j in list(schema[event_type]['参数'].keys()):
                    #print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                    logging.info(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                #print('')
                logging.info('')
            
            f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
            all_f1.append(f1)
    print(f"{event_type}的准确率为：{np.mean(all_f1)}")
    logging.info(f"{event_type}的准确率为：{np.mean(all_f1)}")
    return all_f1

def single_zero_shot(opt,event_type,model):
    
    if not os.path.exists(path):
        os.makedirs(path)

    
    schema_emb=[]
    type2label={}
    k=0
    for key,value in schema['事件'][event_type]['参数'].items():
        inputs=model.tokenizer(key+value,return_tensors="pt")
        for key1 in inputs:
            inputs[key1]=torch.tensor(inputs[key1]).cuda()
        arg_emb=model.bert(**inputs)[0].mean(1).squeeze(0)
        schema_emb.append(arg_emb)
        type2label[key]=k
        k+=1
    schema_emb=torch.stack(schema_emb,0)
    types=list(schema['事件'][event_type]['参数'].keys())
    print(event_type,types,schema_emb.shape)

    
    test_data=[json.loads(line) for line in open(data_path+'/val.json', encoding='utf8')]
    all_f1=[]
    for instance in tqdm(test_data):
        #print('sen_emb',sen_emb.shape)
        for event in instance['event']:
            label=[]
            if event['type']==event_type:
                #print(event)
                span=[]
                if not event['args']:
                    continue
                print('##########################################################\n')
                print(f"{event_type}:{instance['text']}")
                print('')

                logging.info('##########################################################\n')
                logging.info(f"{event['type']}:{instance['text']}")
                logging.info('')

                word,mask,seg=model.tokenize(instance['tokens'])
                word=word.unsqueeze(0).cuda()
                sen_emb=model.bert(word)[0].squeeze(0)

                for cal in event['args']:
                    index=cal['offset']
                    cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                    #print(len(index),cal_emb.shape,opt.offset)
                    cal_emb=cal_emb.mean(0)
                    span.append(cal_emb)
                    label.append(type2label[cal['type']])
                span=torch.stack(span,0)#候选,(M,d)
                logits=similarity(schema_emb.unsqueeze(0),span.unsqueeze(1),-1,opt.sim)#(B,N,M)
                _, pred = torch.max(logits, -1) #(N,1)
                #print(label,pred)
                label=torch.tensor(label)
                for i,cal in enumerate(event['args']):
                #print(schema_emb[event_type]['type'])
                
                    print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}"))
                    for j in types:
                        
                        print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                        logging.info(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                    print('')
                    logging.info('')

                f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
                all_f1.append(f1)
    logging.info(f"{event_type}的准确率为：{np.mean(all_f1)}")
    return all_f1
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', default='中标', help='event type')
    parser.add_argument('--max_length', default=256,type=int, help='max length')
    parser.add_argument('--batch_size', default=32, type=int,help='batch size')
    parser.add_argument('--offset', default=2, type=int,help='计算候选span周围的偏移的embedding')
    parser.add_argument('--seed', default=666, type=int,help='seed')
    parser.add_argument('--sim', default='ou', type=str,help='计算句子和描述的相似度')
    parser.add_argument('--train', default='zero', type=str,help='zero/few')
    parser.add_argument('--tag', default='', type=str)

    parser.add_argument('--lr', default=2e-5, type=float,help='lr')
    parser.add_argument('--epoch', default=100, type=int,help='epoch')




    opt = parser.parse_args()
    return opt

def single_zero_shot_new1(opt,event_type,model):
    #比较cal和模板
    test_data=json.load(open(f'{data_path}/post_val.json', encoding='utf8'))
    all_f1=[]
    type2label={}
    for index,arg_type in enumerate(schema[event_type]['参数'].keys()):
        type2label[arg_type]=index
    types=list(schema[event_type]['参数'].keys())
    print(type2label)
    for data in tqdm(test_data):
        if data['type']==event_type:
            #print('##########################################################\n')
            #print(f"{event_type}:{data['text']}")
            #print('')
            
            logging.info('##########################################################\n')
            logging.info(f"{event_type}:{data['text']}")
            logging.info('')

            word,mask,seg=model.tokenize(data['tokens'])
            word=word.unsqueeze(0).cuda()
            sen_emb=model.bert(word)[0].squeeze(0)
            arg_embs=[]
            
            label=[]
            span=[]
            for cal in data['args']:
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                cal_emb=cal_emb.mean(0)
                span.append(cal_emb)
                for arg_type,arg_text in schema[event_type]['参数'].items():
                    arg_text=arg_text.replace('<M>',cal['text'])
                    inputs=model.tokenizer(arg_text,return_tensors="pt")
                    for key1 in inputs:
                        inputs[key1]=torch.tensor(inputs[key1]).cuda()
                    arg_emb=model.bert(**inputs)[0].mean(1).squeeze(0)
                    arg_embs.append(arg_emb)
                label.append(type2label[cal['type']])
                
            
            arg_embs=torch.stack(arg_embs,0)#候选,(M,d)
            span=torch.stack(span,0)#候选,(M,d)
               
            arg_embs=arg_embs.reshape(len(data['args']),len(schema[event_type]['参数']),768)
            #print(span.shape,arg_embs.shape,label)
            #logits=similarity(arg_embs,sen_emb,-1,opt.sim)#(B,N,M)
            logits=similarity(arg_embs,span.unsqueeze(1),-1,opt.sim)#(B,N,M)
            #print(logits.shape)
            _, pred = torch.max(logits, -1)
            label=torch.tensor(label)
            #print(label,pred)
            for i,cal in enumerate( data['args']):
                #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                logging.info((f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}"))
                for j in list(schema[event_type]['参数'].keys()):
                    #print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                    logging.info(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                #print('')
                logging.info('')

            f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
            all_f1.append(f1)
    print(f"{event_type}的准确率为：{np.mean(all_f1)}")
    logging.info(f"{event_type}的准确率为：{np.mean(all_f1)}")
    return all_f1


def single_zero_shot_new(opt,event_type,model):
    #比较候选词和候选词
    test_data=json.load(open(f'{data_path}/post_val.json', encoding='utf8'))
    all_f1=[]
    type2label={}
    for index,arg_type in enumerate(schema[event_type]['参数'].keys()):
        type2label[arg_type]=index
    types=list(schema[event_type]['参数'].keys())
    print(type2label)
    for data in tqdm(test_data):
        if data['type']==event_type:
            #print('##########################################################\n')
            #print(f"{event_type}:{data['text']}")
            #print('')
            
            logging.info('##########################################################\n')
            logging.info(f"{event_type}:{data['text']}")
            logging.info('')

            word,mask,seg=model.tokenize(data['tokens'])
            word=word.unsqueeze(0).cuda()
            sen_emb=model.bert(word)[0].squeeze(0)
            arg_embs=[]
            
            label=[]
            span=[]
            for cal in data['args']:
                index=cal['offset']
                cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                if len(cal_emb)==0:
                    cal_emb=sen_emb
                cal_emb=cal_emb.mean(0)
                span.append(cal_emb)
                for arg_type,arg_text in schema[event_type]['参数'].items():
                    #arg_text=arg_text.replace('<M>',cal['text'])
                    inputs,begin,end=model.prompt_tokenize(arg_text,cal['text'])
                    inputs=inputs.cuda().unsqueeze(0)
                    arg_emb=model.bert(inputs)[0].squeeze(0)[begin:end].mean(0)
                    arg_embs.append(arg_emb)
                label.append(type2label[cal['type']])
                
            
            arg_embs=torch.stack(arg_embs,0)#候选,(M,d)
            span=torch.stack(span,0)#候选,(M,d)
               
            arg_embs=arg_embs.reshape(len(data['args']),len(schema[event_type]['参数']),768)
            #print(span.shape,arg_embs.shape,label)
            #logits=similarity(arg_embs,sen_emb,-1,opt.sim)#(B,N,M)
            logits=similarity(arg_embs,span.unsqueeze(1),-1,opt.sim)#(B,N,M)
            #print(logits.shape)
            _, pred = torch.max(logits, -1)
            label=torch.tensor(label)
            #print(label,pred)
            for i,cal in enumerate( data['args']):
                #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                logging.info((f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}"))
                for j in list(schema[event_type]['参数'].keys()):
                    #print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                    logging.info(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                #print('')
                logging.info('')

            f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
            all_f1.append(f1)
    print(f"{event_type}的准确率为：{np.mean(all_f1)}")
    logging.info(f"{event_type}的准确率为：{np.mean(all_f1)}")
    return all_f1

def single_zero_shot(opt,event_type,model):
    
    if not os.path.exists(path):
        os.makedirs(path)

    
    schema_emb=[]
    type2label={}
    k=0
    for key,value in schema['事件'][event_type]['参数'].items():
        inputs=model.tokenizer(key+value,return_tensors="pt")
        for key1 in inputs:
            inputs[key1]=torch.tensor(inputs[key1]).cuda()
        arg_emb=model.bert(**inputs)[0].mean(1).squeeze(0)
        schema_emb.append(arg_emb)
        type2label[key]=k
        k+=1
    schema_emb=torch.stack(schema_emb,0)
    types=list(schema['事件'][event_type]['参数'].keys())
    print(event_type,types,schema_emb.shape)

    
    test_data=[json.loads(line) for line in open(data_path+'/val.json', encoding='utf8')]
    all_f1=[]
    for instance in tqdm(test_data):
        #print('sen_emb',sen_emb.shape)
        for event in instance['event']:
            label=[]
            if event['type']==event_type:
                #print(event)
                span=[]
                if not event['args']:
                    continue
                print('##########################################################\n')
                print(f"{event_type}:{instance['text']}")
                print('')

                logging.info('##########################################################\n')
                logging.info(f"{event['type']}:{instance['text']}")
                logging.info('')

                word,mask,seg=model.tokenize(instance['tokens'])
                word=word.unsqueeze(0).cuda()
                sen_emb=model.bert(word)[0].squeeze(0)

                for cal in event['args']:
                    index=cal['offset']
                    cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                    #print(len(index),cal_emb.shape,opt.offset)
                    cal_emb=cal_emb.mean(0)
                    span.append(cal_emb)
                    label.append(type2label[cal['type']])
                span=torch.stack(span,0)#候选,(M,d)
                logits=similarity(schema_emb.unsqueeze(0),span.unsqueeze(1),-1,opt.sim)#(B,N,M)
                _, pred = torch.max(logits, -1) #(N,1)
                #print(label,pred)
                label=torch.tensor(label)
                for i,cal in enumerate(event['args']):
                #print(schema_emb[event_type]['type'])
                
                    print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}"))
                    for j in types:
                        
                        print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                        logging.info(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                    print('')
                    logging.info('')

                f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
                all_f1.append(f1)
    logging.info(f"{event_type}的准确率为：{np.mean(all_f1)}")
    return all_f1

def zero_shot(opt,model):
    model.eval()
    schema_emb={}
    for event_type in schema['事件'].keys():
        schema_emb[event_type]={}
        schema_emb[event_type]['emb']=[]
        type2label={}
        k=0
        for key,value in schema['事件'][event_type]['参数'].items():
            inputs=model.tokenizer(key+value,return_tensors="pt")
            for key1 in inputs:
                inputs[key1]=torch.tensor(inputs[key1]).cuda()
            arg_emb=model.bert(**inputs)[0].mean(1).squeeze(0)
            schema_emb[event_type]['emb'].append(arg_emb)
            type2label[key]=k
            k+=1
        schema_emb[event_type]['emb']=torch.stack(schema_emb[event_type]['emb'],0)
        schema_emb[event_type]['type2label']=type2label
    test_data=json.load(open(data_path+'/post_val.json', encoding='utf8'))
    all_f1={}
    for instance in tqdm(test_data):
        word,mask,seg=model.tokenize(instance['tokens'])
        word=word.unsqueeze(0).cuda()
        sen_emb=model.bert(word)[0].squeeze(0)
        label=[]
        if instance['type'] not in all_f1:
            all_f1[instance['type']]=[]

        span=[]
        type2label=schema_emb[instance['type']]['type2label']
        arg_emb=schema_emb[instance['type']]['emb']
        for cal in instance['args']:
            index=cal['offset']
            cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
            cal_emb=cal_emb.mean(0)
            span.append(cal_emb)
            label.append(type2label[cal['type']])
        span=torch.stack(span,0)#候选,(M,d)
        '''
                logits1=torch.zeros((len(span),len(arg_emb))).cuda()
                for i,v in enumerate(span):
                    for j,va in enumerate(arg_emb):
                        logits1[i][j]=similarity(v,va,-1,opt.sim)
        '''
        logits=similarity(arg_emb.unsqueeze(0),span.unsqueeze(1),-1,opt.sim)#(B,N,M)
        _, pred = torch.max(logits, -1) #(N,1)
        label=torch.tensor(label)
        f1=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor))
        #f1=f1_score(np.array(label),pred.cpu().numpy(), average='macro')
        all_f1[instance['type']].append(f1.item())
        
            
    f1=[]
    for event_type in schema['事件'].keys():       
        f1+=all_f1[event_type]
        print(f'{event_type}准确率为{np.mean(all_f1[event_type])}')
        logging.info(f'{event_type}准确率为{np.mean(all_f1[event_type])}')
    print('总的准确率为',np.mean(f1))
    logging.info(f'总的准确率为{np.mean(f1)}')
    return np.mean(f1)

def few_shot(opt,model):
    train_data=get_loader('train',model,opt.batch_size)
    print(len(train_data))
    #optimer
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(model.parameters(),lr=opt.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300,num_training_steps=opt.epoch*(len(train_data)))
    start_epoch=0
    best_acc=0

    
    if os.path.exists(f'{path}/epoch_last.pth.tar'):
        print('load参数')
        state_dict = torch.load(f'{path}/epoch_last.pth.tar')
        own_state = model.state_dict()
        for name, param in state_dict['state_dict'].items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            #print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        start_epoch = state_dict['epoch']
        acc = state_dict['max_acc']
    model.train()


    
    aves_keys = ['acc', 'loss']
    aves = {k: util.AverageMeter() for k in aves_keys}
    for epoch in range(start_epoch,opt.epoch):
        for data,args in tqdm(train_data):
            schema_emb={}
            for event_type in schema['事件'].keys():
                schema_emb[event_type]={}
                schema_emb[event_type]['emb']=[]
                type2label={}
                k=0
                for key,value in schema['事件'][event_type]['参数'].items():
                    inputs=model.tokenizer(key+value,return_tensors="pt")
                    for key1 in inputs:
                        inputs[key1]=torch.tensor(inputs[key1]).cuda()
                    arg_emb=model.bert(**inputs)[0].mean(1).squeeze(0)
                    schema_emb[event_type]['emb'].append(arg_emb)
                    type2label[key]=k
                    k+=1
                schema_emb[event_type]['emb']=torch.stack(schema_emb[event_type]['emb'],0)
                schema_emb[event_type]['type2label']=type2label

            #编码data
            for key in data:
                data[key]=data[key].cuda()
                #print(data[key].shape) #(b,max_length,d)
            sen_embs=model(data)
            all_label=[]
            all_pred=[]
            loss=0
            for i,sen_emb in enumerate(sen_embs):
                label=[]
                span=[]
                type2label=schema_emb[args['label'][i]]['type2label']
                arg_emb=schema_emb[args['label'][i]]['emb']
                for cal in args['cal_args'][i]:
                    index=cal['offset']
                    cal_emb=sen_emb[max(0,index[0]-opt.offset):min(opt.max_length,index[-1]+opt.offset)]
                    if len(cal_emb)==0:
                        cal_emb=sen_emb[0]
                    else:
                        cal_emb=cal_emb.mean(0)
                    span.append(cal_emb)
                    label.append(type2label[cal['type']])
                span=torch.stack(span,0)#候选,(M,d)
                logits=similarity(arg_emb.unsqueeze(0),span.unsqueeze(1),-1,opt.sim)#(B,N,M)
                _, pred = torch.max(logits, -1) #(N,1)
                all_pred+=pred
                #loss_tmp=model.margin_loss(logits,torch.tensor(label).cuda())
                loss_tmp=model.loss1(logits,torch.tensor(label).cuda())
                loss+=loss_tmp
            label=torch.tensor(label)
            acc=torch.mean((pred.cpu().view(-1) == label.view(-1)).type(torch.FloatTensor)) 
            aves['acc'].update(acc.item(), 1)
            aves['loss'].update(loss.item(), 1)
            #print(loss.item())
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
     
        
        acc=zero_shot(opt,model)
        logging.info(f"epoch{epoch}训练的loss: {aves['loss'].item()},acc:{aves['acc'].item()}|验证acc:{acc}")
        print(f"epoch{epoch}训练的loss: {aves['loss'].item()},acc:{aves['acc'].item()}|验证acc:{acc}")
        model.train()
        ckpt = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'acc': acc,
                    'max_acc': max(best_acc, acc),
                }
        if acc>best_acc:
            print('Best checkpoint', epoch, acc)
            logging.info(f'Best checkpoint{epoch},{acc}')
            torch.save(ckpt, f'{path}/max.pth.tar')
            best_acc=acc
        torch.save(ckpt, f'{path}/epoch_last.pth.tar')
        



def process_data(type):
    test_data=[json.loads(line) for line in open(f'{data_path}/{type}.json', encoding='utf8')]
    post_data=[]
    for data in tqdm(test_data):
        for event in data['event']:
            tmp={}
            if not event['args']:
                continue
            tmp['text']=data['text']
            tmp['tokens']=data['tokens']
            tmp['type']=event['type']
            tmp['trigger']=event['text']
            tmp['offset']=event['offset']
            tmp['args']=event['args']
            post_data.append(tmp)
            
    with open(f'{data_path}/post_{type}.json', 'w',encoding='utf8') as f:
            json.dump(post_data, f,ensure_ascii=False)
    
        





    
    



if __name__ == "__main__":

    cpu_num = 1  # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    opt = get_parser()
    # 设计随机种子
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    path = f"checkpoint/{opt.train}/{opt.sim}"
    data_path='data/DUEE_FIN_LITE'

    schema_folder = 'data/seen_schema'
    schema_file = os.path.join(schema_folder, f'金融信息.yaml')
    schema = load_definition_schema_file(schema_file)
    if not os.path.exists(path):
        os.makedirs(path)

    pretrain_ckpt = 'pretrain/FinBert'
    model = BERTSentenceEncoder(pretrain_ckpt, opt.max_length)
    model.cuda()


    log_name = f"{path}/{opt.sim}{opt.tag}.log"
    logging.basicConfig(level=logging.INFO, filename=log_name,filemode='a')
    logging.info(opt)

    
    if opt.train=='zero':
        f1=[]
        #zero_shot(opt)
        for event_type in schema['事件'].keys():
            f1+=single_zero_shot(opt,event_type,model)
        logging.info(f"总的准确率为：{np.mean(f1)}")
        print(f"总的准确率为：{np.mean(f1)}")
    elif opt.train=='cal_cal':
        schema_file = os.path.join(schema_folder, f'金融信息-new.yaml')
        schema = load_definition_schema_file(schema_file)
        f1=[]
        #zero_shot(opt)
        for event_type in schema.keys():
            f1+=single_zero_shot_new(opt,event_type,model)
        logging.info(f"总的准确率为：{np.mean(f1)}")
        print(f"总的准确率为：{np.mean(f1)}")
    elif opt.train=='cal_cal_type':
        schema_file = os.path.join(schema_folder, f'金融信息_new1.yaml')
        schema = load_definition_schema_file(schema_file)
        f1=[]
        #zero_shot(opt)
        for event_type in schema.keys():
            f1+=cal_cal_type(opt,event_type,model)
        logging.info(f"总的准确率为：{np.mean(f1)}")
        print(f"总的准确率为：{np.mean(f1)}")
    elif opt.train=='cal_tem':
        schema_file = os.path.join(schema_folder, f'金融信息-new.yaml')
        schema = load_definition_schema_file(schema_file)
        f1=[]
        #zero_shot(opt)
        for event_type in schema.keys():
            f1+=single_zero_shot_new1(opt,event_type,model)
        logging.info(f"总的准确率为：{np.mean(f1)}")
        print(f"总的准确率为：{np.mean(f1)}")

    elif opt.train=='all_zero':
        zero_shot(opt,model)

    elif opt.train=='few':
        few_shot(opt,model)
    else:
        #process_data('train')
        #process_data('val')
        print(span2type('2020年11月',model))
        print(span2type('20%',model))
        print(span2type('500$我不熟悉',model))
        print(span2type('华东师范大学',model))
        print(span2type('习近平',model))
        print(span2type('20000股份',model))
