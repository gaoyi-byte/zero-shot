from collections import defaultdict
from copy import deepcopy
from typing import Dict, List
import sys

def tuple_offset(offset):
    if isinstance(offset, tuple):
        return offset
    else:
        return tuple(offset)


def warning_tp_increment(gold, pred, prefix):
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Gold Offset: {gold['offset']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Pred Offset: {pred['offset']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Gold String: {gold['string']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Pred String: {pred['string']}\n")
    sys.stderr.write(f"===============\n")


class Metric:
    """ Tuple Metric """

    def __init__(self, verbose=False, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    def __repr__(self) -> str:
        return f"tp: {self.tp}, gold: {self.gold_num}, pred: {self.pred_num}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {
            prefix + 'tp': tp,
            prefix + 'gold': gold_num,
            prefix + 'pred': pred_num,
            prefix + 'P': p * 100,
            prefix + 'R': r * 100,
            prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
        }

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len( gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)

            if len(gold_list) > 0 and len(pred_list) > 0:
                # guarantee length same
                assert len(gold_list[0]) == len(pred_list[0])

            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    if self.match_mode == 'normal':
                        # Each Gold Instance can be matched one time
                        dup_gold_list.remove(pred)

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)


class EventScorer():
    def __init__(self, verbose=False, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}
        self.role_metrics = {
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
        }

    def load_sen_list(self,sen_list):
        """[summary]
        Args:
            sen_list (List[List[Dict]]): List of Sentece, each sentence contains a List of Event Dict
                [
                    { # Sentance Event Record
                        'type': 'Die',
                        'cal_args': [
                            {'type': 'Victim', 'offset': [17], 'text': 'himself'},
                            {'type': 'Agent', 'offset': [5, 6], 'text': 'John Joseph'},
                            {'type': 'Place', 'offset': [23], 'text': 'court'}
                        ]
                        'gold_args': [
                            {'type': 'Victim', 'offset': [17], 'text': 'himself'},
                            {'type': 'Agent', 'offset': [5, 6], 'text': 'John Joseph'},
                            {'type': 'Place', 'offset': [23], 'text': 'court'}
                        ]
                    },
                ]
        Returns:
            List[Dict]: List of Sentece, each sentence contains Four List of Event Tuple
                [
                    {
                        
                        'offset_role': [('Die', 'Victim', (17,)), ('Die', 'Agent', (5, 6)), ('Die', 'Place', (23,))],
                        'string_role': [('Die', 'Victim', 'himself'), ('Die', 'Agent', 'John Joseph'), ('Die', 'Place', 'court')]
                    },
                    ...
                ]
        """
        gold_list = []
        pred_list = []
        for sen in sen_list:
            instance = defaultdict(list)
            for arg in sen['gold_args']:
                instance['offset_role'] += [(
                    sen['label'], arg['type'],
                    tuple_offset(arg['offset']))]
                instance['string_role'] += [
                    (sen['label'], arg['type'], arg['text'])
                ]
            gold_list += [instance]
            instance = defaultdict(list)
            for arg in sen['cal_args']:
                if arg['pred_type']:#如果有预测类型
                    instance['offset_role'] += [(
                        sen['label'], arg['pred_type'],
                        tuple_offset(arg['offset']))]
                    instance['string_role'] += [
                        (sen['label'], arg['pred_type'], arg['text'])
                    ]
            pred_list += [instance]
        return gold_list,pred_list

    def eval_instance_list(self,gold_instance_list,
                           pred_instance_list,
                           verbose=False,
                           match_mode='normal'):
        """[summary]
        Args:
            gold_instance_list (List[Dict]): List of Sentece, each sentence contains Four List of Event Tuple
                [
                    {
                        
                        'offset_role': [('Die', 'Victim', (17,)), ('Die', 'Agent', (5, 6)), ('Die', 'Place', (23,))],
                        'string_role': [('Die', 'Victim', 'himself'), ('Die', 'Agent', 'John Joseph'), ('Die', 'Place', 'court')]
                    },
                    ...
                ]
            pred_instance_list (List[Dict]): List of Sentece, each sentence contains four List (offset, string) X (trigger, role) of Event List
                [
                    {
                        
                        'offset_role': [('Attack', 'Attacker', (5, 6)), ('Attack', 'Place', (23,)), ('Attack', 'Target', (17,))],
                        'string_role': [('Attack', 'Attacker', 'John Joseph'), ('Attack', 'Place', 'court'), ('Attack', 'Target', 'himself')],
                    },
                    ...
                ]
            verbose (bool, optional): [description]. Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal`.
        Returns:
            Dict: Result of Evaluation
                (offset, string) X (trigger, role) X (gold, pred, tp, P, R, F1)
        """
        
        

        for pred, gold in zip(pred_instance_list, gold_instance_list):
            pre_string_tp, pre_offset_tp = self.role_metrics[
                'string'].tp, self.role_metrics['offset'].tp

            for eval_key in self.role_metrics:
                self.role_metrics[eval_key].count_instance(
                    gold_list=gold.get(eval_key + '_role', []),
                    pred_list=pred.get(eval_key + '_role', []))

            post_string_tp, post_offset_tp = self.role_metrics[
                'string'].tp, self.role_metrics['offset'].tp
            if verbose and post_offset_tp - pre_offset_tp != post_string_tp - pre_string_tp:
                warning_tp_increment(gold=gold, pred=pred, prefix='Role')

        results = dict()
        for eval_key in self.role_metrics:
            results.update(self.role_metrics[eval_key].compute_f1(eval_key+'-'))
            

        return results