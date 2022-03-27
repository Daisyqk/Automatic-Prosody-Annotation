import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import PositionalEncoder
from Sublayers import Norm
import copy
from SubSampleCNN import VGG2L
from transformers import BertModel
from embedding import PositionalEncoding


class Prosody_extrator(nn.Module):
    def __init__(self,  d_text, checkpoint, num_tags, seq_len):
        super().__init__()

        self.bert = BertModel.from_pretrained(checkpoint)
        self.linear_1 = nn.Linear(d_text, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.length = seq_len

    def forward(self, input_ids, attention_mask, labels):
        embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask ,output_hidden_states=True).last_hidden_state  #  (B, 512, 768)
        bert_out = self.linear_1(embeddings) #(b, 512, 3)
        loss_mask = attention_mask.bool() #b, 512
        temp_mask = torch.ones(attention_mask.size()).cuda() - attention_mask  # (B，512)
        masked_pre = bert_out.masked_fill(temp_mask.unsqueeze(-1).bool(), value=torch.tensor(0))# (B，length, 3)
        bert_loss = F.cross_entropy(masked_pre[loss_mask], labels[loss_mask])
        attention_mask_1 = attention_mask.byte()
        crf_loss = -self.crf(emissions=bert_out, tags=labels.long(), mask=attention_mask_1) / torch.sum(attention_mask)

        loss = bert_loss + crf_loss

        crf_pre = self.crf.decode(emissions=bert_out, mask=attention_mask_1)
        crf_pre = [[p[j] if (j > 0 and j < len(p) - 1) else 0 for j in range(0, self.length)] for p in crf_pre]
        crf_pre = torch.as_tensor(crf_pre).cuda()
        return loss, crf_pre

    
def get_model(opt):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Prosody_extrator(d_text=opt.d_text, checkpoint=opt.bert_checkpoint, num_tags=opt.num_tags, seq_len=opt.bert_embedding_length)

    return model
