
from copy import deepcopy
from encoder import BERTSentenceEncoder
from data_loader import get_loader
import util
from tqdm import tqdm
import sys
import torch
import numpy as np
import json
import re
import os
import logging
import os
import random
import yaml
import warnings
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')
from model import dep,tem,prompt,cal_ver_emb,span2type,similarity
from score import EventScorer

def load_definition_schema_file(filename):
    """Load schema file in Yaml
    读取 YAML 定义的 Schema 文件
    """
    return yaml.load(open(filename, encoding='utf8'), Loader=yaml.FullLoader)

def rank(origin_emb,args,arguments):
    event=args['label']
    single_schema=schema_tem[event]['参数']
    tems=schema_tem[event]['格式'].split('^')#每个要素的以^分割
    #print(tems)
    arg2tem={}
    for tem in tems:
        key=re.findall(r"<.*>",tem)[0][1:-1]
        #print(key,tem)
        arg2tem[key]=tem.strip()
    sentence=['']#最后的句子
    for key,value in arg2tem.items():
        if key not in arguments:
            continue
        else:
            flag=1
            tmp=[]
            for arg in arguments[key]:
                for sen in sentence:
                    tmp.append(sen+value.replace(f'<{key}>',arg))
            sentence=deepcopy(tmp)
    data = {'word': [], 'mask': [], 'seg': []}
    for sen in sentence:
        word, mask,seg = encoder.tokenize(sen)
        data['word'].append(word)
        data['mask'].append(mask)
        data['seg'].append(seg)
    data['word']= torch.stack(data['word'], 0).cuda()
    data['mask']= torch.stack(data['mask'], 0).cuda()
    data['seg']= torch.stack(data['seg'], 0).cuda()
    with torch.no_grad():
        sen_embs=[]
        tmp={}
        for i in range(0,len(data['word']),320):
            #print(i,i+250)
            for key in data:
                tmp[key]=data[key][i:i+320]
            sen_emb=encoder(tmp).mean(1)#(sen,D)
            sen_embs.append(sen_emb)
            #print(sen_emb.shape)
        #sen_embs= torch.stack(sen_embs, 0).cuda()
        #sen_embs=sen_embs.reshape(len(sentence),-1)
        sen_embs=torch.cat(sen_embs,0)
        #print(sen_embs.shape,len(sentence))
        assert len(sentence)==sen_embs.shape[0]
        logits=similarity(origin_emb.mean(0),sen_embs,-1,opt.sim)#(B,N,M)
        #print(origin_emb.mean(0).shape,sen_embs.mean(1).shape,logits.shape,)
        _, pred = torch.max(logits, -1)
    #print(pred)
    max_sen=sentence[pred.item()]
    #print(max_sen)
    #print(args['gold_args'])
    for cal in args['cal_args']:
        if cal['text'] not in max_sen:
            cal['pred_type']=None
        #print(cal)
    
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
    score=EventScorer(match_mode='set')
    
    for data,args in tqdm(test_data):
        for key in data:
            data[key]=data[key].cuda()
        with torch.no_grad():
            
            sen_embs,logits,label=model(data,args,mask=True)#tuple(b*logits)
        for b in range(len(logits)):
            type2label=model.schema_info[args[b]['label']]['type2label']
            types=model.schema_info[args[b]['label']]['types']
            args_span={}
            if logits[b]==[]:
                continue
            _, pred = torch.max(logits[b], -1) #(N,1)
            arguments={}
            for i,cal in enumerate(args[b]['cal_args']):
                pred_id=pred[i]
                if opt.test=='0':
                    pred_label=types[pred_id]
                    cal['pred_type']=pred_label
                elif '1' in opt.test:
                    if logits[b][i][pred_id]==0:
                        print('#####')
                        cal['pred_type']=None
                        continue  
                    else:
                        pred_label=types[pred_id]
                        cal['pred_type']=pred_label
                if '2' in opt.test:
                    if pred_label in args_span:
                        pre_cal_id=args_span[pred_label]
                        if logits[b][i][pred_id]>logits[b][pre_cal_id][pred_id]:
                            args[b]['cal_args'][pre_cal_id]['pred_type']=None
                            args_span[pred_label]=i
                        else:
                            cal['pred_type']=None
                    else:
                        args_span[pred_label]=i
                if '3' in opt.test:
                    if pred_label in arguments:
                        arguments[pred_label].append(cal['text'])
                    else:
                        arguments[pred_label]=[cal['text']]
                    rank(sen_embs[b],args[b],arguments)
            
            #rank(sen_embs[b],args[b],arguments)
            #sys.exit()

            if opt.log_error:
                logging.info('##########################################################\n')
                logging.info(f"{args[b]['label']}:{args[b]['text']}")
                logging.info('')
                
                logging.info('真实span的标签')
                for i,cal in enumerate(args[b]['gold_args']):
                    #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']} 实体类型：{cal['type']}"))
                
                
                logging.info('预测span的标签:')
                for i,cal in enumerate(args[b]['cal_args']):
                    if not cal['pred_type']:
                        logging.info(f"{cal['text']} 实体类型：{cal['span_type']},reject")
                        continue
                    #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']} 实体类型：{cal['span_type']} |预测标签是：{cal['pred_type']}"))
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
    
    train_data=get_loader('train',opt.model,encoder,opt.batch_size,opt.k,event_types,schema_tem)
    val_data=get_loader('val',opt.model,encoder,opt.batch_size,opt.k,event_types,schema_tem)
    test_data=get_loader('test',opt.model,encoder,opt.batch_size,opt.k,event_types,schema_tem)
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
    if opt.model=='cls':
        model_path=os.path.join(path,f'{event_types[0]}/model')
    else:
        model_path=os.path.join(path,'model')

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
            print(name)
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
            if opt.model=='dep':
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
        logging.info(f"[Train] epoch{epoch}: result:{result},loss:{all_loss.item()}")#验证
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

def zero_shot_newdata(opt,model,event_types,test_data,ckpt=None):
    model.eval()
    model.update_schema()
    print('test',len(test_data))
    #type_emb,span_types=cal_ver_emb(model.encoder)
    score=EventScorer(match_mode='set')
    
    for data,args in tqdm(test_data):
        for key in data:
            data[key]=data[key].cuda()
        with torch.no_grad():
            
            sen_embs,logits,label=model(data,args,mask=True)#tuple(b*logits)
        for b in range(len(logits)):
            type2label=model.schema_info[args[b]['label']]['type2label']
            types=model.schema_info[args[b]['label']]['types']
            args_span={}
            if logits[b]==[]:
                continue
            _, pred = torch.max(logits[b], -1) #(N,1)
            arguments={}
            for i,cal in enumerate(args[b]['cal_args']):
                pred_id=pred[i]
                if opt.test=='0':
                    pred_label=types[pred_id]
                    cal['pred_type']=pred_label
                elif '1' in opt.test:
                    if logits[b][i][pred_id]==0:
                        print('#####')
                        cal['pred_type']=None
                        continue  
                    else:
                        pred_label=types[pred_id]
                        cal['pred_type']=pred_label
                if '2' in opt.test:
                    if pred_label in args_span:
                        pre_cal_id=args_span[pred_label]
                        if logits[b][i][pred_id]>logits[b][pre_cal_id][pred_id]:
                            args[b]['cal_args'][pre_cal_id]['pred_type']=None
                            args_span[pred_label]=i
                        else:
                            cal['pred_type']=None
                    else:
                        args_span[pred_label]=i
                if '3' in opt.test:
                    if pred_label in arguments:
                        arguments[pred_label].append(cal['text'])
                    else:
                        arguments[pred_label]=[cal['text']]
                    rank(sen_embs[b],args[b],arguments)
            
            #rank(sen_embs[b],args[b],arguments)
            #sys.exit()

            if opt.log_error:
                logging.info('##########################################################\n')
                logging.info(f"{args[b]['label']}:{args[b]['text']}")
                logging.info('')
                
                logging.info('真实span的标签')
                for i,cal in enumerate(args[b]['gold_args']):
                    #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']} 实体类型：{cal['type']}"))
                
                
                logging.info('预测span的标签:')
                for i,cal in enumerate(args[b]['cal_args']):
                    if not cal['pred_type']:
                        logging.info(f"{cal['text']} 实体类型：{cal['span_type']},reject")
                        continue
                    #print(f"{cal['text']}  的gold标签是：{cal['type']},预测标签是：{types[pred[i]]},与gold的sim是：{logits[i][label[i]]},与预测的sim是：{logits[i][pred[i]]}")
                    logging.info((f"{cal['text']} 实体类型：{cal['span_type']} |预测标签是：{cal['pred_type']}"))
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

    path = f"checkpoint/{opt.train}/{opt.model}"
    if not os.path.exists(path):
        os.makedirs(path)
    data_path='data/DUEE_FIN_LITE'


    schema_folder = 'data/seen_schema'
    pretrain_ckpt ='pretrain/FinBert'
    log_name = f"{path}/{opt.sim}{opt.tag}.log"

    if opt.log_error:
        log_name = f"{path}/{opt.sim}{opt.test}{opt.tag}-error.log"
    if opt.train=='few':
        log_name = f"{path}/{opt.sim}{opt.tag}-{opt.k}.log"
        opt.weight=True
    
        
    logging.basicConfig(level=logging.INFO, filename=log_name)
    logging.info(opt)

    schema_file = os.path.join(schema_folder, f'金融信息.yaml')
    schema_dep = load_definition_schema_file(schema_file)
    schema_file = os.path.join(schema_folder, f'金融_tem.json')
    schema_tem=json.load(open(schema_file, encoding='utf8'))

     

    
    if opt.model=='dep':
        encoder = BERTSentenceEncoder(pretrain_ckpt, opt.max_length,opt.train)
        model=dep(encoder,schema_dep,schema_tem,opt.sim,opt.offset,opt.weight)
        model.cuda()
    elif opt.model=='tem':
        encoder = BERTSentenceEncoder(pretrain_ckpt, opt.max_length,opt.train)
        model=tem(encoder,schema_dep,schema_tem,opt.sim,opt.offset,opt.weight)
        model.cuda()
    elif opt.model=='prompt':
        encoder = BERTSentenceEncoder(pretrain_ckpt, opt.max_length,opt.train)
        model=prompt(encoder,schema_dep,schema_tem,opt.sim,opt.offset)
        model.cuda()
    else:
        raise NotImplementedError
    event_types=list(schema_tem.keys())
    if opt.train=='zero':
        if opt.log_error:
            for event_type in event_types:
                test_data=get_loader('test',opt.model,encoder,opt.batch_size,opt.k,[event_type],schema_tem)
                zero_shot(opt,model,[event_type],test_data)      
        else:
            for event_type in event_types:
                test_data=get_loader('test',opt.model,encoder,opt.batch_size,opt.k,[event_type],schema_tem)
                zero_shot(opt,model,[event_type],test_data)
    elif opt.train=='few':
        few_shot(opt,model,event_types)
        
    else:
        raise NotImplementedError 
    


