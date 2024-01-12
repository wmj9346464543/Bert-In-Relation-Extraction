#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : main function to carry out training and testing
@Author             : Kevinpro
@version            : 1.0
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
# from transformers import BertPreTrainedModel
from transformers import BertModel
from model import BERT_Classifier
from loader import load_train
from loader import load_dev
from loader import map_id_rel
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(44)
rel2id, id2rel = map_id_rel()
print(len(rel2id), id2rel)

def load_data_and_transform(file_path):
    # 加载数据
    data = load_train(file_path) if 'train' in file_path else load_dev(file_path)
    # 转换为numpy数组并构建tensor
    text = torch.tensor(np.stack([t.numpy() for t in data['text']]))
    mask = torch.tensor(np.stack([t.numpy() for t in data['mask']]))
    label = torch.tensor(data['label'])
    # 打印数据形状
    print("data shapes:", text.shape, mask.shape, label.shape)
    return torch.utils.data.TensorDataset(text, mask, label)

# 加载和处理数据
train_file_path = "D:/code/nlp/nlp-examples-main/data/DUIE/train.json"
dev_file_path = "D:/code/nlp/nlp-examples-main/data/DUIE/dev.json"

train_dataset = load_data_and_transform(train_file_path)
dev_dataset = load_data_and_transform(dev_file_path)
print('data down!')

def eval(net, dataset, batch_size):
    net.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        iter = 0
        for text, mask, y in train_iter:
            iter += 1
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)
            text = text.to(device)
            mask = mask.to(device)
            y = y.to(device)

            outputs = net(text, mask, y)
            # print(y)
            loss, logits = outputs[0], outputs[1]
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
            s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
        acc = (1.0 * correct.numpy()) / total
        print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
        return acc

def train(net, dataset, num_epochs, learning_rate, batch_size):
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0)
    # optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre = 0.95
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        iter = 0
        for text, mask, y in train_iter:
            iter += 1
            optimizer.zero_grad()
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)
            text = text.to(device)
            mask = mask.to(device)
            y = y.to(device)
            # print(text.shape)
            loss, logits = net(text, mask, y)
            # print(y)
            # print(loss.shape)
            # print("predicted",predicted)
            # print("answer", y)
            loss.backward()
            optimizer.step()
            # print(outputs[1].shape)
            # print(output)
            # print(outputs[1])
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss = loss.detach().cpu()
        print("epoch ", str(epoch), " loss: ", loss.mean().numpy().tolist(), "right", correct.cpu().numpy().tolist(),
              "total", total, "Acc:", correct.cpu().numpy().tolist() / total)
        acc = eval(model, dev_dataset, batch_size)
        if acc > pre:
            pre = acc
            torch.save(model, str(acc) + '.pth')
    return


model = BERT_Classifier(label_num=len(rel2id))  # model=nn.DataParallel(model,device_ids=[0,1])
model = model.to(device)
num_epochs = 300
learning_rate = 0.002
train_batch_size = 128
train(model, train_dataset, num_epochs, learning_rate, train_batch_size)
# 最后在百度DuIE数据集的完整测试集上达到95.37%正确率