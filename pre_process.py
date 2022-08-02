import logging
import re
from encoder import BERTSentenceEncoder
from data_loader import get_loader
import util
import torch
import numpy as np
import json
import os
import random
import yaml
import warnings
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')

all_arg_span_type=['PER','ORG','TIME','T-NUM','T-NUM-MONEY','T-NUM-PERCENT',"T-OFFICER",'V-股票','V-股份']
all_arg_span_type=set(all_arg_span_type)
def load_definition_schema_file(filename):
    """Load schema file in Yaml
    读取 YAML 定义的 Schema 文件
    """
    return yaml.load(open(filename, encoding='utf8'), Loader=yaml.FullLoader)
       
def process_schema(schema):
    for event_type in schema:
        num=[]
        for arg_type,arg_text in schema[event_type]['参数'].items():
            tmp=arg_text.split('$')

            arg_span_type=tmp[0]
            #arg_span_type=arg_span_type.replace('T-ORG','T-O')
            #arg_span_type=arg_span_type.replace('T-PER','T-O')
            #arg_span_type=arg_span_type.replace('T-OFFICE','T-O')
            arg_span_type=arg_span_type.split(',')
            print(arg_span_type)
            for i in range(len(arg_span_type)):
                arg_span_type[i]=arg_span_type[i].strip()
            arg_span_type=set(arg_span_type)
            if '^T-O'in arg_span_type:
                arg_span_type.remove('^T-O')
            arg_tmp_text=tmp[1].split('|')
            schema[event_type]['参数'][arg_type]={}
            schema[event_type]['参数'][arg_type]['span_type']=list(arg_span_type)
            schema[event_type]['参数'][arg_type]['tem']=arg_tmp_text
            num.append(len(arg_tmp_text))
        schema[event_type]['tem_num']=num
    print(schema)
    with open(f'data/seen_schema/金融_tem.json', 'w',encoding='utf8') as f:
        json.dump(schema, f,ensure_ascii=False)
    return schema

def process_data(data_path):
    data = json.load(open(f'{data_path}/class_train.json', encoding='utf8'))
    test_data = json.load(open(f'{data_path}/class_val.json', encoding='utf8'))
    train_data={}
    val_data={}
    for key,value in data.items():
        num=len(test_data[key])
        train_data[key]=data[key][:-num]
        val_data[key]=data[key][-num:]
    with open(f'{data_path}/train_train.json', 'w',encoding='utf8') as f:
        json.dump(train_data, f,ensure_ascii=False)
    with open(f'{data_path}/train_val.json', 'w',encoding='utf8') as f:
        json.dump(val_data, f,ensure_ascii=False)

def process_gold_span(data_path):
    train_data = json.load(open(f'{data_path}/train_train.json', encoding='utf8'))
    test_data = json.load(open(f'{data_path}/class_val.json', encoding='utf8'))
    val_data = json.load(open(f'{data_path}/train_val.json', encoding='utf8'))
    for key,value in test_data.items():
        print(key,len(train_data[key]),len(val_data[key]),len(test_data[key]))
        for data in value:
            for cal in data['args']:
                #cos_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'cos')
                #ou_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'ou')
                #dot_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'dot')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-DATE','TIME')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-DATE','TIME')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-ORG','ORG')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-PER','PER')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
                cal['ou_span_type']=cal['ou_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
                cal['dot_span_type']=cal['dot_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
    for key,value in val_data.items():
        for data in value:
            for cal in data['args']:
                #cos_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'cos')
                #ou_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'ou')
                #dot_type=span2type(cal['text'],data['text'],encoder,type_emb,span_types,prompt_tem,'dot')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-DATE','TIME')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-ORG','ORG')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-PER','PER')
                cal['cos_span_type']=cal['cos_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
                cal['ou_span_type']=cal['ou_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
                cal['dot_span_type']=cal['dot_span_type'].replace('T-NUM-IN-PERCENT','T-NUM-PERCENT')
    with open(f'{data_path}/train_val.json', 'w',encoding='utf8') as f:
        json.dump(val_data, f,ensure_ascii=False)
    with open(f'{data_path}/class_val.json', 'w',encoding='utf8') as f:
        json.dump(test_data, f,ensure_ascii=False)

def process_cal_span(data_path,types,schema):
    if types=='train':
        pro_data = json.load(open(f'{data_path}/train_train.json', encoding='utf8'))
    elif types=='test':
        pro_data = json.load(open(f'{data_path}/class_val.json', encoding='utf8'))
    else:
        pro_data = json.load(open(f'{data_path}/train_val.json', encoding='utf8'))

    texts=[]
    for key in schema:
        for data in pro_data[key]:
            data['text']=data['text'].replace('\n','。')
            texts.append(data['text'])
            
    from LAC import LAC

    # 装载LAC模型
    lac = LAC(mode='lac')
    # 批量样本输入, 输入为多个句子组成的list，平均速率更快
    lac_result = lac.run(texts)
    print(len(lac_result))
    k=0
    for key in schema:
        all_type=[]
        #计算该类别的type
        for arg,arg_type in schema[key]['参数'].items():
            print(key,arg_type['span_type'])
            if 'MISC' in arg_type['span_type']:
                all_type=all_arg_span_type
                break
            if '^'in arg_type['span_type'][0]:
                exclude=set()
                for s_span_type in arg_type['span_type']:
                    exclude.add(s_span_type[1:])
                all_type+=(all_arg_span_type-exclude)
            else:
                all_type+=arg_type['span_type']
        all_type=set(all_type)
        

        value=pro_data[key]
        for data in value:
            data['cal_args']=[]
            unque_span=set()
            offset=0
            #抽取PER,ORG,TIME
            for i in range(1,len(lac_result[k][0])):
                if lac_result[k][1][i]==lac_result[k][1][i-1]:#两个一样，合并成一个span
                    lac_result[k][1][i]='none'
                    lac_result[k][0][i-1]=lac_result[k][0][i-1]+lac_result[k][0][i]

            for span,span_type in zip(lac_result[k][0],lac_result[k][1]):
                res={}
                if span_type in all_type and span not in ''.join(unque_span):
                    res['text']=span
                    #unque_span.add(span)
                    res['span_type']=span_type
                    res['offset']=list(range(offset,offset+len(span)))
                    data['cal_args'].append(res)
                offset+=len(span) 
            k+=1

            #抽取数字信息（金钱，百分数，股份数量）
            res_span=re.finditer(r"[千数]?[\d亿万]+\.?[\d,]*[亿万w]?[美港欧]?[元%股]",data['text'])
            for i in res_span:
                res={}
                text=data['text'][i.span()[0]:i.span()[1]]
                res['text']=text
                res['offset']=list(range(i.span()[0],i.span()[1]))
                if '元' in text and 'T-NUM-MONEY' in all_type:
                    res['span_type']='T-NUM-MONEY'
                elif '%' in text and 'T-NUM-PERCENT' in all_type:
                    res['span_type']='T-NUM-PERCENT'
                elif '股' in text and 'T-NUM' in all_type:
                    res['span_type']='T-NUM'
                    res['text']=text[:-1]
                    res['offset']=list(range(i.span()[0],i.span()[1]-1))
                else:
                    break
                #unque_span.add(text)
                data['cal_args'].append(res)
            #抽取特定词表
            if 'V-股票' or 'V-股份' in all_type:
                res_span=re.finditer(r"质押股份|质押股票",data['text'])
                for i in res_span:
                    res={}
                    text=data['text'][i.span()[0]:i.span()[1]]
                    res['text']=text[-2:]
                    res['offset']=list(range(i.span()[0]+2,i.span()[1]))
                    res['span_type']='V-股票'
                    data['cal_args'].append(res)
                    break
            #某些特定类型，无法抽取直接赋值
            for cal in data['args']:
                if cal['type'] in ['高管职位','变动类型','变动后职位']:
                    cal['span_type']='T-OFFICER'
                    data['cal_args'].append(cal)
           
    with open(f'data/event3-repeat-{types}.json', 'w',encoding='utf8') as f:
       json.dump(pro_data, f,ensure_ascii=False)

def process_type_data(data_path,types):
    if types=='train':
        pro_data = json.load(open(f'{data_path}/train_train.json', encoding='utf8'))
    elif types=='test':
        pro_data = json.load(open(f'{data_path}/class_val.json', encoding='utf8'))
    else:
        pro_data = json.load(open(f'{data_path}/train_val.json', encoding='utf8'))
    
    texts=[]
    for key,value in pro_data.items():
        for data in value:
            data['text']=data['text'].replace('\n','。')
            texts.append(data['text'])
            
    from LAC import LAC

    # 装载LAC模型
    lac = LAC(mode='lac')
    # 批量样本输入, 输入为多个句子组成的list，平均速率更快
    lac_result = lac.run(texts)
    k=0
    for key,value in pro_data.items():
        for data in value:
            data['cal_args']=[]
            unque_span=set()
            offset=0
            #处理单个句子信息
            for i in range(1,len(lac_result[k][0])):
                if lac_result[k][1][i]==lac_result[k][1][i-1]:#两个一样，合并成一个span
                    lac_result[k][1][i]='none'
                    lac_result[k][0][i-1]=lac_result[k][0][i-1]+lac_result[k][0][i]

            for span,span_type in zip(lac_result[k][0],lac_result[k][1]):
                res={}
                if span_type in ['PER','ORG','TIME'] and span not in ''.join(unque_span):
                    res['text']=span
                    unque_span.add(span)
                    res['span_type']=span_type
                    res['offset']=list(range(offset,offset+len(span)))
                    data['cal_args'].append(res)
                offset+=len(span)
            k+=1
            
    with open(f'data/{types}.json', 'w',encoding='utf8') as f:
        json.dump(pro_data, f,ensure_ascii=False)
                    
def process_num_data(data_path,types):
    
    pro_data = json.load(open(f'data/{types}.json', encoding='utf8'))
    
    texts=[]
    for key,value in pro_data.items():
        for data in value:
            #处理数字
            res_span=re.finditer(r"[千数]?[\d亿万]+\.?[\d,]*[亿万w]?[美港欧]?[元%股]",data['text'])
            unque_span=set()
            for i in res_span:
                res={}
                text=data['text'][i.span()[0]:i.span()[1]]
                res['text']=text
                res['offset']=list(range(i.span()[0],i.span()[1]))
                if '元' in text:
                    res['span_type']='T-NUM-MONEY'
                elif '%' in text:
                    res['span_type']='T-NUM-PERCENT'
                elif '股' in text:
                    res['span_type']='T-NUM'
                    res['text']=text[:-1]
                    res['offset']=list(range(i.span()[0],i.span()[1]-1))
                unque_span.add(text)
                data['cal_args'].append(res)
            res_span=re.finditer(r"质押股份|质押股票",data['text'])
            for i in res_span:
                res={}
                text=data['text'][i.span()[0]:i.span()[1]]
                res['text']=text[-2:]
                res['offset']=list(range(i.span()[0]+2,i.span()[1]))
                res['span_type']='V-股票'
                data['cal_args'].append(res)
                break
            for cal in data['args']:
                if cal['type'] in ['高管职位','变动类型','变动后职位']:
                    cal['span_type']='T-OFFICER'
                    data['cal_args'].append(cal)

            
    with open(f'data/{types}.json', 'w',encoding='utf8') as f:
        json.dump(pro_data, f,ensure_ascii=False)

def compute(gold_list,pred_list):
    tp=0
    for pred_str in pred_list:
        for gold_str in gold_list:
            if set(gold_str) & set(pred_str) :
                print(gold_str,pred_str)
                tp+=1
                break
    return tp
    
def log_error():
    pro_data = json.load(open(f'data/event3-repeat-test.json', encoding='utf8'))
    res={'NER':0,'RE':0,'GOLD':0}
    pred=0
    gold=0
    tp=0
    for key in ['质押','中标','高管变动']:
        items=pro_data[key]   
        for item in items:
            logging.info(f"type:{item['type']},text:{item['text']}") 
            cal_span=set()  
            for cal in item['args']:
                gold+=1
                cal_span.add(cal['text'])
            logging.info(f'gold_span:{cal_span}')
            logging.info('抽取出来的span有:')
            pred_span=set()
            for cal in item['cal_args']:
                pred+=1
                if cal['text'] in cal_span:
                    if cal['span_type'] in ['PER','ORG','TIME']:
                        ee_type='NER'
                    elif cal['span_type'] in ['T-NUM','T-NUM-MONEY','T-NUM-PERCENT']:
                        ee_type='RE'
                    else:
                        ee_type='GOLD'
                    res[ee_type]+=1
                    tp+=1
                    logging.info(f"\t {cal['text']},抽取方法：{ee_type}")
                    cal_span.remove(cal['text'])
                else:
                    pred_span.add(cal['text'])
            
            logging.info(f'没有抽取出来的:{cal_span}')
            logging.info(f'多抽取出来的:{pred_span}')
            
    logging.info(f'{res}')
    logging.info(f'gold:{gold},pred{pred},tp:{tp}')


def eval():
    tp,gold,pred=0,0,0
    test_data = json.load(open(f'data/event3-test.json', encoding='utf8'))
    for key in ['质押','中标','高管变动']:
        value=test_data[key]
        for data in value:
            gold_list,pred_list=[],[]
            gold_offset ,pred_offset=[],[]
            for cal in data['args']:
                gold_list.append(cal['text'])
                gold_offset.append(cal['offset'])
            for cal in data['cal_args']:
                pred_list.append(cal['text'])
                pred_offset.append(cal['offset'])

            gold_list = set(gold_list)
            pred_list = set(pred_list)
            

            gold += len(gold_list)
            pred += len(pred_list)
            tp1=len( gold_list & pred_list)
            #tp2 =compute(gold_offset,pred_offset)
            #print(tp2,tp1)
            tp+=tp1
    p=tp/pred
    r=tp/gold
    f1=2*p*r/(p+r)
    print(tp,gold,pred,p,r,f1)
    #1161 1656 3182 0.36486486486486486 0.7010869565217391 0.47995039272426626

def process_schema1():
    schema_file = os.path.join('data/seen_schema', f'金融_tem.json')
    schema=json.load(open(schema_file, encoding='utf8'))
    for event_type in schema:
        maxl=0
        for key in schema[event_type]['参数']:
            maxl=max(maxl,len(key))
        schema[event_type]['maxl']=maxl
    with open(f'data/seen_schema/金融_tem.json', 'w',encoding='utf8') as f:
        json.dump(schema, f,ensure_ascii=False)
    return schema

if __name__ == "__main__":

    cpu_num = 1  # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    opt = util.get_parser()
    # 设计随机种子
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    data_path='data/DUEE_FIN_LITE'

    schema_folder = 'data/seen_schema'
    schema_file = os.path.join(schema_folder, f'金融_tem.json')
    schema=json.load(open(schema_file, encoding='utf8'))
    log_name = "/home/yigao/data/ee/checkpoint/process/ee.log"
    
    logging.basicConfig(level=logging.INFO, filename=log_name,filemode = 'w')
    logging.info(opt)
    
    
    '''
    process_cal_span(data_path,'train',schema)
    process_cal_span(data_path,'dev',schema)
    process_cal_span(data_path,'test',schema)
    #'''
    
    process_gold_span(data_path)
    #eval()
    
    


