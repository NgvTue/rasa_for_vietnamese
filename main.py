import os,time
import typing
from typing import Any, Optional, Text, Dict, List, Type
from torchcrf import CRF
import numpy as np
import torch
from torch import nn as nn
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
text = "tôi đi học hôm nay nhé minflow 10 năm qua, Quảng Ninh và Đà Nẵng gần như thay nhau chiếm Chỉ số năng lực cạnh tranh cấp tỉnh là một trong những chỉ báo đánh giá khả năng xây dựng môi trường kinh doanh"
tokens = text.split()
print(len(tokens))
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


class ModelJoin(nn.Module):
    def __init__(self, pretrained_bert, tokenizer, intents, tag2id, dropout = 0.3, use_crf = True, ignore_index = -100):
        super().__init__()
        self.bert = pretrained_bert
        self.tokenizer = tokenizer
        self.intents = intents
        self.tag2id = tag2id
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.bert.config.hidden_size,
                len(self.intents)
            )
        )

        self.slot_detection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.bert.config.hidden_size,
                len(self.tag2id)
            )
        )
        self.use_crf = use_crf
        self.ignore_index = ignore_index
        if use_crf:
            self.crf = CRF(num_tags=len(self.tag2id), batch_first=True)

    def forward(self, input_ids, **kwargs):
        outputs = self.bert(
            input_ids, 
            **kwargs
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        # print(sequence_output.shape,pooled_output.shape)
        # pooled_output = torch.cat((sequence_output, pooled_output), 1)
        pooled_output = torch.mean(sequence_output, 1) + pooled_output
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_detection(sequence_output)

        return intent_logits, slot_logits

    
    def forward_train(self, 
                      inputs_ids,
                      attention_mask,
                      intent_label_ids,
                      slot_labels_ids,
                      slot_loss_coef = 1.):

        intent_logits, slot_logits = self.forward(
            inputs_ids,
            attention_mask=attention_mask
        )        

        total_loss = 0
        # 1. Intent Softmax
        
        intent_loss_fct = nn.CrossEntropyLoss()
        intent_loss = intent_loss_fct(intent_logits.view(-1, len(self.intents)), intent_label_ids.view(-1))

        total_loss += intent_loss
        # print(intent_loss)

        # 2. Slot Softmax
        
        if self.use_crf:            
            slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss  # negative log-likelihood
            
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, len(self.tag2id))[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, len(self.tag2id)), slot_labels_ids.view(-1))
        total_loss += slot_loss_coef * slot_loss
        if total_loss < 0 :
          raise ValueError(
              slot_logits,
              slot_labels_ids,
              slot_loss,
              intent_loss
          )
        outputs = ((intent_logits, slot_logits),)  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs 

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

intents = {0: 'ask_ability', 1: 'concept', 2: 'decry', 3: 'goodbye', 4: 'greet', 5: 'praise', 6: 'prevent', 7: 'reason', 8: 'symptom', 9: 'thankyou', 10: 'treatment'}
ids_2_tags = {1: 'B-thalassemia_syn', 2: 'I-thalassemia_syn',
                 3: 'L-thalassemia_syn', 4: 'U-thalassemia_syn', 5: 'B-ung_thư_gan_syn', 6: 'I-ung_thư_gan_syn',
                 7: 'L-ung_thư_gan_syn', 8: 'U-ung_thư_gan_syn', 9: 'B-ung_thư_trực_tràng_syn',
                 10: 'I-ung_thư_trực_tràng_syn', 11: 'L-ung_thư_trực_tràng_syn',
                 12: 'U-ung_thư_trực_tràng_syn', 0: 'O'}
tags_2_ids = {
    v:k for k,v in ids_2_tags.items()
}

models = ModelJoin(
    phobert, tokenizer,
    intents,
    tags_2_ids
)
models = models
print(np.sum([i.numel() for i in models.parameters()]))
inputs_ids = torch.randn(
    35,21
) * 100 
inputs_ids = inputs_ids - torch.min(inputs_ids)
inputs_ids = inputs_ids.to(torch.long)
print(inputs_ids)
attention_mask = torch.ones(35, 21).to(torch.long)
intent_label_ids = torch.ones(
    35,1
).to(torch.long)
st = time.time()
slot_labels_ids = torch.ones(35,21).to(torch.long)
with torch.no_grad():
    outs = models.forward_train(
        inputs_ids,
        attention_mask,
        intent_label_ids,
        slot_labels_ids,
    )

end_ = time.time()
print(f"take {end_ - st}")

