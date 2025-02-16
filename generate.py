import os
import argparse
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from pandas import DataFrame
from llm import query
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from llm import query
import pandas as pd
import re
import string

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0
#Please convert the triples from chains of thought into the second paragraph.
def llm_gen(select1,select2,question):
    prompt='''
    Here are some chains of thought. The chains are '''+select2+\
    '''
    Please convert each triple into a sentence and connect the sentences into a paragraph.
    And here are some triples from knowledge graph. The triplets are '''+select1+\
    '''
     Please convert each triple into a sentence and connect the sentences into a paragraph.
    The question is '''+ question+\
    '''
    Please generate as many answers as possible for the question based on the generated paragraph and sentences.
    Let's think step by step. 
    Only return the answers. If there are more than one answer, please use \'|\' to split them.'''
    model='llama-3.1-8b-instant'
    ans=query(prompt, model)
    return ans

def gen_eva(args):
    input_dir1 = os.path.join('dataset', args.d)
    dataset1 = load_dataset('parquet', data_dir=input_dir1, split=args.split)
    input_dir2 = os.path.join('dataset/select', args.d, args.split+'select.pkl')
    dataset2 = pd.read_pickle(input_dir2)
    df1 = dataset1.to_pandas()
    #df2 = dataset2.to_pandas()
    dataset=pd.merge(df1, dataset2, on='id')

    true=[]
    k=df1['answer']
    #print(len(k))
    true=list(k.astype(str))

    print(dataset.columns)
    
    pred=[]
    pbar = tqdm(total=len(dataset),desc='generate')
    for i in range(len(dataset)):
        select1=[]
        select2=[]
        #if len(dataset.loc[i,'answer_x'])==1: continue
        for k in dataset.loc[i,'select1']:   #add KG
            select1.append(k[1])
            #select1.append(k['t'])
        for k in dataset.loc[i,'ind']:      #add CoT
            select2.append(k)

        select1=' '.join(select1)
        select2=' '.join(select2)
        ans=''
        while(1):
            try:
                ans=llm_gen(select1,select2,dataset.loc[i,'question_x'])
            except Exception as e:
                print("API is error")
            if len(ans)!=0: break
        #ans=llm_gen(select1,select2,select3,dataset.loc[i,'question_x'])
        pred.append(ans)
        pbar.update(1)
    pbar.close()

    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    pbar = tqdm(total=len(pred),desc='evaluate')
    for prediction, answer in zip(pred,true):
        prediction = prediction.replace("|", "\n")
        answer = answer.split(" ")
        prediction = prediction.split("\n")

        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)
        pbar.update(1)
    pbar.close()

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    #f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)
    f1= 2 * pre * recall / (pre + recall)
    

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--data_path', type=str, default='dataset/cot_index')
    argparser.add_argument('--d', '-d', type=str, default='rog-webqsp')
    argparser.add_argument('--split', type=str, default='test')
    argparser.add_argument('--output_path', type=str, default='dataset/select')
   # argparser.add_argument('--undirected', action='store_true', help='whether the graph is undirected')
    
    args = argparser.parse_args()

    gen_eva(args)
