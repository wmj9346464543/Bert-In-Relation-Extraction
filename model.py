#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : Model defined here. Use BERT as BackBone
@Author             : Kevinpro
@version            : 1.0
'''
from transformers import BertModel
import torch.nn as nn
class BERT_Classifier(nn.Module):
    def __init__(self,label_num):
        super().__init__()
        pretrained_file = r"D:\code\pretrained_models\bert-base-chinese"
        self.encoder = BertModel.from_pretrained(pretrained_file)
        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, attention_mask ,label=None):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        if label == None:
            return None,x
        else:
            return self.criterion(x,label),x
