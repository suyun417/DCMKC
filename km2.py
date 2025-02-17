import networkx as nx
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset,load_from_disk
from multiprocessing import Pool
from pandas import DataFrame
from src.llm import query
from src.buildcot import llm_index1,llm_index2
from transformers import AutoTokenizer, AutoModel
from src.interchains import select_node
import pandas as pd
from difflib import SequenceMatcher
from src.check import checkk
import traceback

def similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def km(left,right):
    G = nx.Graph()
    left_nodes = left     #CoT
    right_nodes=[]
    for bian in right:  #selected KG
        right_nodes.append(bian['t'])
 
    G.add_nodes_from(left_nodes, bipartite=0)  # bipartite=0 left
    G.add_nodes_from(right_nodes, bipartite=1)  # bipartite=1 right

    edges=[]
    for bian in right:
        edge=(bian['h'],bian['t'],{'weight': bian['r']})
        edges.append(edge)
    G.add_edges_from(edges)

    matching = nx.max_weight_matching(G, weight='weight')
    return matching

def iterate(args):
    output_dir = os.path.join(args.output_path, args.d, args.split)
    
    # Load dataset
    input_dir1 = os.path.join('dataset/graph_index', args.d, args.split)
    dataset1 = load_from_disk(input_dir1)
    df1 = dataset1.to_pandas()
    input_file = os.path.join(args.data_path, args.d)
    dataset2 = load_dataset('parquet', data_dir=input_file, split=args.split)
    df2 = dataset2.to_pandas()
    dataset=pd.merge(df1, df2, on='id')
    #dataset.to_pickle(output_dir+'select.pkl')
    index1=[]
    index11=[]
    index12=[]
    indexc=[]
    indexc1=[]
    indexc2=[]
    pbar = tqdm(total=len(dataset),desc='select')
    for i in range(len(dataset)):
        ind_g=[]
        for s1 in dataset.loc[i,'ind_g']: ind_g.append(s1)
        str_g=[]
        for s1 in dataset.loc[i,'str_g']: str_g.append(s1)
        kg=""
        ck=0
        while(1):
            ck=ck+1
            if ck==2:
                matching11=matching1
                cot11=cot
            if ck==3:
                matching12=matching1
                cot12=cot
            if ck==1:
                while(1):   #initial CoT
                    cot=""
                    try:
                        cot=llm_index1(dataset.loc[i,'question'])
                    except Exception as e:
                        traceback.print_exc()
                        print("API is error")
                    if len(cot)!=0: break
            else:
                while(1):   #CoT
                    cot=""
                    try:
                        cot=llm_index2(dataset.loc[i,'question'],kg)
                    except Exception as e:
                        traceback.print_exc()
                        print("API is error")
                    if len(cot)!=0: break

            select=select_node(cot,ind_g,str_g)  #selected KG
            kg1=[]
            matching1=[]

            while(1):
                if len(select)<len(cot): break
                matching=km(cot,select)    #list, the first is CoT node, the second is KG node.
                if len(matching)==0: break
                select,flagg=checkk(matching,select,dataset.loc[i,'link'])
                if flagg==0: continue
                matching1.extend(matching)
                for s1 in matching: kg1.append(s1[1])
           
            kg1=' '.join(kg1)

            if similarity(kg,kg1)>0.8 or ck>=3: break
            kg=kg1
            
        '''index1.append(matching1)
        index2.append(select_fb)
        indexc.append(cot)'''
        index11.append(matching11)
        indexc1.append(cot11)
        index12.append(matching12)
        indexc2.append(cot12)
        index1.append(matching1)
        indexc.append(cot)
        pbar.update(1)
    pbar.close()
    '''dataset['select1']=index1
    dataset['select']=index2
    dataset['ind']=indexc'''
    dataset['select1']=index11
    dataset['ind1']=indexc1
    dataset['select2']=index12
    dataset['ind2']=indexc2
    dataset['select']=index1
    dataset['ind']=indexc
    dataset.to_pickle(output_dir+'select.pkl')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='dataset')
    argparser.add_argument('--d', '-d', type=str, default='rog-webqsp')
    argparser.add_argument('--split', type=str, default='test')
    argparser.add_argument('--output_path', type=str, default='dataset/select')
   # argparser.add_argument('--undirected', action='store_true', help='whether the graph is undirected')
    
    args = argparser.parse_args()

    iterate(args)