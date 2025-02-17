import os
import argparse
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from pandas import DataFrame
from src.llm import query
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def mean_pooling(model_output, attention_mask):
    #First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def index_graph(args):
    input_file = os.path.join(args.data_path, args.d)
   # data_path = f"{args.d}_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.output_path, args.d, args.split)
    # Load dataset
    dataset = load_dataset('parquet', data_dir=input_file, split=args.split)
    dataset1=dataset.select_columns(['id', 'graph'])
    #print(dataset1[0]['graph'][0])
    
    tokenizer = AutoTokenizer.from_pretrained('./sentence-bert')
    model = AutoModel.from_pretrained('./sentence-bert')
    index1=[]
    index2=[]
    index3=[]
    pbar = tqdm(total=len(dataset1))
    for data in dataset1:
        ind=[]
        for tri in data['graph']:
            ans=','.join(tri)
            ind.append(ans)
        link=[]
        for tri1 in data['graph']:
            for tri2 in data['graph']:
                if tri1[0]==tri2[2]:
                    ans1=','.join(tri1)+','.join(tri2)
                    link.append(ans1)
        index1.append(ind)
        index3.append(link)
        encoded_input = tokenizer(ind, padding=True, truncation=True, return_tensors='pt') 
        #print('begin embedding')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()  #embedding
        #print('embedding completed')
        #print(sentence_embeddings[0])
        ss=list(sentence_embeddings)
        #print(ss[0])
        index2.append(ss)
        '''for aaa in range(len(dataset1)-1):
            index1.append(ind)
            index2.append(ss)
        break'''
        pbar.update(1)
    pbar.close()
    
    dataset1=dataset1.add_column('str_g',index1)
    dataset1=dataset1.add_column('ind_g',index2)
    dataset1=dataset1.add_column('link_g',index2)
    #print(dataset1[0]['ind_g'][0])
    dataset1.select_columns(['id','str_g', 'ind_g']).save_to_disk(output_dir)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='dataset')
    argparser.add_argument('--d', '-d', type=str, default='rog-webqsp')
    argparser.add_argument('--split', type=str, default='test')
    argparser.add_argument('--output_path', type=str, default='dataset/graph_index')
   # argparser.add_argument('--undirected', action='store_true', help='whether the graph is undirected')
    
    args = argparser.parse_args()

    index_graph(args)
