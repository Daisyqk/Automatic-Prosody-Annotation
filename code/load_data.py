import pandas as pd
# import torchtext
# from torchtext import data
# from Tokenize import tokenize
# from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
from datasets import load_dataset
from transformers import BertModel, BertTokenizer
from typing import Tuple
import sys
import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset


class read_text():
    def __init__(self, checkpoint, bert_embedding_length):
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.MAX_LENGTH = bert_embedding_length

    def get_tokenize_and_label(self, examples):
        all_text = []
        all_label = []
        all_id = []
        for example in examples['text']:
            contain = eval(example)
            id = contain['id']
            org_text = ''.join(contain['source'])  # str
            label = '0' + (''.join(contain['target'])).replace('#', '') + '0'  # 头尾补0  对应[CLS]和[SEP]
            all_id.append(id)
            all_text.append(org_text)
            all_label.append(list(label))
        tokenized = self.tokenizer(all_text, truncation=True, padding='max_length', max_length=self.MAX_LENGTH)
        tokenized['labels'] = all_label
        tokenized['id'] = all_id
        return tokenized

    def read_text_and_label(self, train_path, test_path):
        if train_path == None:
            dataset = load_dataset('text', data_files={'test': test_path})
            test_dataset = dataset['test'].map(self.get_tokenize_and_label, batched=True)
            train_dataset = None
        else:
            dataset = load_dataset('text', data_files={
            'train': train_path,
            'test': test_path
            })
            train_dataset = dataset['train'].map(self.get_tokenize_and_label, batched=True)
            test_dataset = dataset['test'].map(self.get_tokenize_and_label, batched=True)
        return train_dataset, test_dataset


class Load_audio_and_text_data(Dataset):
    def __init__(self, dataset, bert_embedding_length, loader):
        self.data = dataset
        self.loader = loader
        self.bert_embedding_length = bert_embedding_length
    def __getitem__(self, index):
        attention_mask = self.data[index]['attention_mask']
        input_ids = self.data[index]['input_ids']
        labels = self.data[index]['labels']
        id = self.data[index]['id']
        org_len = len(labels)
        padded_labels = [int(labels[i]) if i < org_len else 0 for i in range(0, self.bert_embedding_length)]
        source = ''.join(eval(self.data[index]['text'])['source'])

        mfcc = self.loader[id]  # sequence_len, 83
        mfcc = torch.as_tensor(mfcc)

        return attention_mask, input_ids, padded_labels, org_len, mfcc, source, id

    def __len__(self):
        return len(self.data)



