import networkx as nx
from tqdm import tqdm
from datasets import load_dataset, Dataset,load_from_disk
from multiprocessing import Pool
from pandas import DataFrame
from llm import query
from buildcot import llm_index1,llm_index2
from transformers import AutoTokenizer, AutoModel
from interchains import select_node
import pandas as pd
from difflib import SequenceMatcher


def checkk(matching,select,link):
    kg=[]
    for s in matching: 
        kg.append(s[1])
    num=0
    for s1 in kg:
        ff=0
        for s2 in kg:  
            ss=s1+s2
            if ss in link:
                ff=1 
        if ff==0:
            select.remove(s1) 
        if ff==1: num=num+1

    if num==len(kg): 
        flag=1
        for pipei in matching:       #remove matched chain
            for jiedian in select:
                if pipei[1]==jiedian['t'] and pipei[0]==jiedian['h']:
                    select.remove(jiedian)
    else: flag=0
    return select,flag
