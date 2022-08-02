import joblib
from encoder import BERTSentenceEncoder
from data_loader import get_loader
import util
from tqdm import tqdm
import torch
import numpy as np
import json
import argparse
import os
import logging
import os
import random
import yaml
import warnings
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')
from model import dep,tem,cls
from score import EventScorer

def load_definition_schema_file(filename):
    """Load schema file in Yaml
    读取 YAML 定义的 Schema 文件
    """
    return yaml.load(open(filename, encoding='utf8'), Loader=yaml.FullLoader)
       
def zero_shot(opt,model,event_types,test_data,ckpt=None):
    if ckpt:
        state_dict = torch.load(ckpt)
        own_state = model.state_dict()
        for name, param in state_dict['state_dict'].items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            own_state[name].copy_(param)
    
    model.eval()
    model.update_schema()
    print('test',len(test_data))
    #type_emb,span_types=cal_ver_emb(model.encoder)
    score=EventScorer()
    
    for data,args in tqdm(test_data):
        for key in data:
            data[key]=data[key].cuda()
        with torch.no_grad():
            _,logits,label=model(data,args,mask=True)#tuple(b*logits)
        for b in range(len(logits)):
            _, pred = torch.max(logits[b], -1) #(N,1)
            for i,cal in enumerate(args[b]['cal_args']):
                cal['pred_type']=model.schema_info[args[b]['label']]['types'][pred[i]]
            if opt.log_error:
                logging.info('##########################################################\n')
                logging.info(f"{args[b]['label']}:{args[b]['text']}")
                logging.info('')
                type2label=model.schema_info[args[b]['label']]['type2label']
                types=model.schema_info[args[b]['label']]['types']
                for i,cal in enumerate(args[b]['cal_args']):
                    #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']} 实体类型：{cal['cos_span_type']} | gold标签是：{cal['type']}|预测标签是：{types[pred[i]]},与gold的sim是：{logits[b][i][label[b][i]]},与预测的sim是：{logits[b][i][pred[i]]}"))
                    #logging.info((f"{cal['text']}  gold标签是：{cal['type']}|预测标签是：{types[pred[i]]},与gold的sim是：{logits[b][i][label[b][i]]},与预测的sim是：{logits[b][i][pred[i]]}"))
                    for j in list(schema_tem[args[b]['label']]['参数'].keys()):
                            #print(f"{cal['text']}与{j}的sim是：{logits[i][type2label[j]]}")
                        logging.info(f"{cal['text']}与{j}的sim是：{logits[b][i][type2label[j]]}")
                        #print('')
                    logging.info('')
        label_list,pred_list=score.load_sen_list(args)
        result=score.eval_instance_list(pred_list,label_list)
    print(f"[EVAL]:{event_types},offset_f1:{result['offset-F1']},string-f1:{result['string-F1']}")
    logging.info(f"[EVAL]:{event_types}, offset_f1:{result['offset-F1']},string-f1:{result['string-F1']},all:{result}")
        
    return result

def few_shot(opt,model,event_types):
    
    train_data=get_loader('train',opt.train,encoder,opt.batch_size,opt.k,event_types,schema_tem)
    val_data=get_loader('val',opt.train,encoder,opt.batch_size,opt.k,event_types,schema_tem)
    test_data=get_loader('test',opt.train,encoder,opt.batch_size,opt.k,event_types,schema_tem)
    print('train',len(train_data))
    #type_emb,span_types=cal_ver_emb(model.encoder)

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
    model_path=os.path.join(path,f'{event_types[0]}/model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    best_f1=0
    if os.path.exists(f'{model_path}/{opt.sim}{opt.tag}-{opt.k}-last.pth'):
        print('load参数')
        state_dict = torch.load(f'{model_path}/{opt.sim}{opt.tag}-{opt.k}-last.pth')
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
        best_f1 = state_dict['max_f1']
    print(start_epoch,opt.epoch)
    model.train()
    
    for epoch in range(start_epoch+1,opt.epoch):
        score=EventScorer()
        all_loss=util.AverageMeter()
        #训练
        for data,args in tqdm(train_data):
            model.update_schema()
            for key in data:
                data[key]=data[key].cuda()
            _,logits,label=model(data,args)#tuple(b*logits)
            loss=0
            for b in range(len(logits)):
                _, pred = torch.max(logits[b], -1) #(N,1)
                loss_tmp=model.loss(logits[b],label[b])
                loss+=loss_tmp
                for i,cal in enumerate(args[b]['cal_args']):
                    cal['pred_type']=model.schema_info[args[b]['label']]['types'][pred[i]]
            loss=loss/(b+1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()    
            label_list,pred_list=score.load_sen_list(args)
            result=score.eval_instance_list(pred_list,label_list)
            all_loss.update(loss.item(), 1)
        print(f"[Train] epoch{epoch}: result:{result},loss:{all_loss.item()}")
        logging.info(f"[Train] epoch{epoch}: result:{result},loss:{all_loss.item()}")
        #验证
        if (epoch+1)%5==0:
            val_result=zero_shot(opt,model,event_types,val_data)
            model.train()
            ckpt = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'f1': val_result['string-F1'],
                        'max_f1': max(best_f1, val_result['string-F1']),
                    }
            if val_result['string-F1']>best_f1:
                print('Best checkpoint', epoch, val_result['string-F1'])
                logging.info(f"Best checkpoint{epoch},{val_result['string-F1']}")
                torch.save(ckpt, f'{model_path}/{opt.sim}{opt.tag}-{opt.k}-max.pth')
                best_f1=val_result['string-F1']
            torch.save(ckpt, f'{model_path}/{opt.sim}{opt.tag}-{opt.k}-last.pth')
    print('testing')
    zero_shot(opt,model,event_types,test_data,f'{model_path}/{opt.sim}{opt.tag}-{opt.k}-max.pth')

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

    path = f"checkpoint/single_{opt.train}/{opt.model}"
    if not os.path.exists(path):
        os.makedirs(path)
    data_path='data/DUEE_FIN_LITE'

    schema_folder = 'data/seen_schema'
    pretrain_ckpt ='pretrain/FinBert'
    log_name = f"{path}/{opt.sim}{opt.tag}.log"

    if opt.log_error:
        log_name = f"{path}/{opt.sim}{opt.tag}-error.log"
    if opt.train=='few':
        log_name = f"{path}/{opt.sim}{opt.tag}-{opt.k}.log"
    
        
    logging.basicConfig(level=logging.INFO, filename=log_name)
    logging.info(opt)

    schema_file = os.path.join(schema_folder, f'金融信息.yaml')
    schema_dep = load_definition_schema_file(schema_file)
    schema_file = os.path.join(schema_folder, f'金融_tem.json')
    schema_tem=json.load(open(schema_file, encoding='utf8'))
    opt.weight=True

     

    encoder = BERTSentenceEncoder(pretrain_ckpt, opt.max_length,opt.train)
    event_types=list(schema_tem.keys())
    if opt.model=='cls':
        for event_type in event_types:
            model=cls(encoder,event_type,schema_dep,schema_tem,opt.sim,opt.offset)
            model.cuda()
            print(f'训练{event_type}ing')
            few_shot(opt,model,[event_type])
    elif opt.model=='cls_dep':
        for event_type in event_types:
            model=cls(encoder,event_type,schema_dep,schema_tem,opt.sim,opt.offset,dep=True)
            model.cuda()
            print(f'训练{event_type}ing')
            few_shot(opt,model,[event_type])
    elif opt.model=='tem':
        for event_type in event_types:
            model=tem(encoder,schema_dep,schema_tem,opt.sim,opt.offset,opt.weight)
            model.cuda()
            print(f'训练{event_type}ing')
            few_shot(opt,model,[event_type])
    elif opt.model=='dep':
        for event_type in event_types:
            model=dep(encoder,schema_dep,schema_tem,opt.sim,opt.offset,opt.weight)
            model.cuda()
            print(f'训练{event_type}ing')
            few_shot(opt,model,[event_type])
    else:
        raise NotImplementedError
    


