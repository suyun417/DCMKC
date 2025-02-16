This code is for the paper "DCMKC: A Dual Consistency Matching Approach for Multi-hop Question Answering in LLMs".

The datasets can be downloaded from huggingface (https://huggingface.co/datasets/rmanluo/RoG-webqsp; https://huggingface.co/datasets/rmanluo/RoG-cwq).  

## Try DCMKC

### 1) Data Preprocessing
```
python buildkg.py -d [dataset name]
```

### 2) Reasoning
```
python km2.py -d [dataset name]
```

### 3) Generate answers and evaluate
```
python generate.py -d [dataset name]
```
You need to change the **key** in llm.py to your own api_key.