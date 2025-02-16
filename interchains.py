import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
from multiprocessing import Pool
from pandas import DataFrame
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.pandas_compat as pd_compat

def compute_sim_score(v1, v2) :
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mean_pooling(model_output, attention_mask):
    #First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def select_node(cot,ind_g,str_g):
    tokenizer = AutoTokenizer.from_pretrained('./sentence-bert')
    model = AutoModel.from_pretrained('./sentence-bert')
    encoded_input = tokenizer(cot, padding=True, truncation=True, return_tensors='pt') 
       
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()  #embedding

    select=[]
    for j1 in range(0,len(sentence_embeddings)):
        li=[]
        for j in range(len(ind_g)):
            s={}
            s['h']=cot[j1]
            s['t']=str_g[j]
            s['r']=compute_sim_score(sentence_embeddings[j1], ind_g[j])
            li.append(s)
        li.sort(key=lambda item: item['r'], reverse=True)

        select.append(li[0])
        if len(li)<2: continue
        select.append(li[1])
        if len(li)<3: continue
        select.append(li[2])
        if len(li)<4: continue
        select.append(li[3])
        if len(li)<5: continue
        select.append(li[4])
        if len(li)<6: continue
        select.append(li[5])
        if len(li)<7: continue
        select.append(li[6])
        if len(li)<8: continue
        select.append(li[7])
    return select
