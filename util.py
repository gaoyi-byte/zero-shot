import time
import numpy as np
import scipy.stats as stats
import json
import argparse
from tqdm import tqdm
def mean_confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  stderr = stats.sem(a)
  h = stderr * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
  return h

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg


class Timer(object):
  def __init__(self):
    self.start()

  def start(self):
    self.v = time.time()

  def end(self):
    return time.time() - self.v

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', default='中标', help='event type')
    parser.add_argument('--max_length', default=512,type=int, help='max length')
    parser.add_argument('--batch_size', default=16, type=int,help='batch size')
    parser.add_argument('--offset', default=10, type=int,help='计算候选span周围的偏移的embedding')
    parser.add_argument('--seed', default=666, type=int,help='seed')
    parser.add_argument('--sim', default='cos', type=str,help='计算句子和描述的相似度')
    parser.add_argument('--model', default='tem', type=str,help='tem/prompt/dep/cls/dep_cls')
    parser.add_argument('--train', default='zero', type=str,help='zero/few')
    parser.add_argument('--test', default='0', type=str,help='0/1/12/13')
    

    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--k', default=10, type=int,help='训练集数量')
    parser.add_argument('--log_error', action='store_true', help='不使用标签')
    parser.add_argument('--weight', action='store_true', help='加入除bert以外的参数')
    parser.add_argument('--lr', default=2e-5, type=float,help='lr')
    parser.add_argument('--epoch', default=100, type=int,help='epoch')

    opt = parser.parse_args()
    return opt



def process_data(data_path,type):
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

def process_class_data(data_path,type):
    test_data=[json.loads(line) for line in open(f'{data_path}/{type}.json', encoding='utf8')]
    data_class={}
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
            if event['type'] not in data_class:
              data_class[event['type']]=[]
            data_class[event['type']].append(tmp)
            
    with open(f'{data_path}/class_{type}.json', 'w',encoding='utf8') as f:
            json.dump(data_class, f,ensure_ascii=False)