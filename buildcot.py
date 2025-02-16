import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from pandas import DataFrame
from llm import query
import time

def llm_index1(question):
    prompt='Generate the thought chains step by step for the following question. The question is'+ question+\
'''Maybe there are two or more thought chains, please generate them all.
And here are some examples of sentences converted into triples.

sentence: The Cayman Islands was affected by the cyclone known as Hurricane Paloma.
triplet: Cayman Islands, meteorology.cyclone_affected_area.cyclones, Hurricane Paloma

sentence: Alexander Bustamante was born in Hanover Parish.
triplet: Hanover Parish, location.location.people_born_here, Alexander Bustamante

Please convert each generated sentence in the thought chains into a triplet following the examples and return them.        
Only return the triplets without any number. '''
    model='llama-3.1-8b-instant'
    ans1=query(prompt, model)
    ans1=ans1.split(':')[-1]
    ans1=ans1.split('\n')
    ans = list(filter(lambda x: x is not None and x != "" and x != [] and x != {} and x != 0 and ',' in x, ans1))
    #print(ans)
    return ans

def llm_index2(question,kg):
    prompt='Here are some triplets from knowledge graph. The triplets are '+kg+\
    '''Please generate the thought chains step by step for the following question based on the triplets from knowledge graph.
    The question is'''+ question+\
    '''Maybe there are two or more thought chains, please generate them all.
    And here are some examples of sentences converted into triples.

sentence: The Cayman Islands was affected by the cyclone known as Hurricane Paloma.
triplet: Cayman Islands, meteorology.cyclone_affected_area.cyclones, Hurricane Paloma

sentence: Alexander Bustamante was born in Hanover Parish.
triplet: Hanover Parish, location.location.people_born_here, Alexander Bustamante

Please convert each generated sentence in the thought chains into a triplet following the examples and return them.        
Only return the triplets without any number. '''
    model='llama-3.1-8b-instant'
    ans1=query(prompt, model)
    ans1=ans1.split(':')[-1]
    ans1=ans1.split('\n')
    ans = list(filter(lambda x: x is not None and x != "" and x != [] and x != {} and x != 0 and ',' in x, ans1))
    #print(ans)
    return ans