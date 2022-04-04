import torch
from torch.nn.utils.rnn import pad_sequence
def my_collate(batch):
    attention_mask = []
    input_ids = []
    padded_labels = []
    org_len = []
    mfcc = []
    source = []
    id = []
    for b in batch:
        attention_mask.append(b[0])
        input_ids.append(b[1])
        padded_labels.append(b[2])
        org_len.append(b[3])
        mfcc.append(b[4])
        source.append(b[5])
        id.append(b[6])
    mfcc = pad_sequence(mfcc, batch_first=True)
    attention_mask = torch.tensor(attention_mask)
    input_ids = torch.tensor(input_ids)
    org_len = torch.tensor(org_len)
    padded_labels = torch.tensor(padded_labels)
    mfcc = torch.as_tensor(mfcc)
    return attention_mask, input_ids, padded_labels, org_len, mfcc, source, id


