import os,time
import typing
from typing import Any, Optional, Text, Dict, List, Type

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
def load_model(path=None):
    if path is None:
        path = 'vinai/phobert-base'
    phobert = AutoModel.from_pretrained(path)
    # For transformers v4.x+: 
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return phobert, tokenizer
device = 'cpu'
st_t = time.time()
phobert, tokenizer = load_model()
import time
end_t = time.time()
print("load done ", end_t-st_t  )
st_t = time.time()
text = "tôi đi học hôm nay nhé minflow "
tokens = text.split()
cache_word = []
len_word = []
unk_token = tokenizer.unk_token
sep_token = tokenizer.sep_token
cls_token = tokenizer.cls_token
for token in tokens:
    word_tokens = tokenizer.tokenize(token)
    if not word_tokens:
        word_tokens = [unk_token]
    cache_word = cache_word + word_tokens
    len_word.append(
        len(word_tokens)
    )
cache_word += [sep_token]
cache_word = [cls_token] + cache_word
# encoder done
tensor_inputs = tokenizer.convert_tokens_to_ids(cache_word)
tensor_inputs = torch.tensor(
    tensor_inputs, dtype=torch.long
).unsqueeze(0)
tensor_inputs = tensor_inputs.to(device)
end_t = time.time()
print("done ", end_t-st_t)
st_t = time.time()
with torch.no_grad():
    out_vectors = phobert(tensor_inputs)
end_t = time.time()
print("predict done", end_t - st_t)
word_feature = out_vectors[0].squeeze(0).to("cpu").numpy()

sentence_feature = out_vectors[1].squeeze(0).to("cpu").numpy()
st = 0
word_reduce = []
for i in range(len(len_word)):
    l = len_word[i]
    en = st + l
    word_reduce.append(
        np.mean(word_feature[st:en,...], axis = 0)
    )
    st = en

word_reduce = np.stack(word_reduce, axis = 0)
st_t = time.time()
print(word_reduce.shape, sentence_feature.shape)
print("done ",st_t-end_t)