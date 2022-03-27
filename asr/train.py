import argparse
import os
import time
import torch
from Models import get_model
import torch.nn.functional as F
from Optim import CosineWithRestarts
from load_data import read_text
from load_data import Load_audio_and_text_data
from torch.utils.data import DataLoader
import numpy as np
import kaldiio
from my_collate import my_collate
from predict_result import predict_result
from calculate_pr import calculate_pr
from espnet_local.nets.pytorch_backend.nets_utils import make_non_pad_mask
import pytorch_warmup as warmup
import random


def train_model(model, opt):

    print("training model...")

    for epoch in range(opt.epochs):
        total_loss = 0
        mean_loss = 0
        c_m = torch.zeros(5, 5)
        train_sum = []
        train_loss = []


        model.train()
        for i, batch in enumerate(opt.train): 

            attention_mask, input_ids, padded_labels, org_len, mfcc, source, id = batch   #size:batch, length
            attention_mask = attention_mask.cuda()
            input_ids = input_ids.cuda()
            padded_labels = padded_labels.cuda()
            mfcc = mfcc.cuda()

            mfcc_len = [mfcc[i].size(0) for i in range(0, mfcc.size(0))]
            src_mask = make_non_pad_mask(mfcc_len).unsqueeze(-2).cuda()

            preds, _ = model(mfcc=mfcc, input_ids=input_ids, attention_mask=attention_mask, mfcc_mask=None, src_mask=src_mask)  #(B，length, 3)

            opt.optimizer.zero_grad()

            loss_mask = attention_mask.bool()
            temp_mask = torch.ones(attention_mask.size()).cuda() - attention_mask
            masked_pre = preds.masked_fill(temp_mask.unsqueeze(-1).bool(), value=torch.tensor(0))
            masked_pred_label = torch.argmax(torch.softmax(input=masked_pre, dim=-1), dim=-1)

            loss = F.cross_entropy(masked_pre[loss_mask], padded_labels[loss_mask])
            loss.backward()

            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=20, norm_type=2.0)
            opt.optimizer.step()
            opt.lr_scheduler.step(opt.lr_scheduler.last_epoch + 1)
            opt.warmup_scheduler.dampen()

            mean_loss += loss.item()
            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                avg_loss = mean_loss/opt.printevery
                print("epoch: %d    batch: %d   loss: %.3f" % (epoch, i, avg_loss))
                mean_loss = 0
            padded_labels = padded_labels.int()
            masked_pred_label = masked_pred_label.int()
            for b in range(0, padded_labels.size(0)):
                l = padded_labels[b]
                p = masked_pred_label[b]
                for id in range(0, len(l)):
                    label = l[id]
                    pred = p[id]
                    c_m[label, pred] += 1

        train_loss.append(round((total_loss / (i + 1)), 3))
        print("epoch %d loss on train set : avg_loss: %.3f " % (epoch, total_loss / (i + 1)))
        print("comfusion_matrix")
        print(c_m)
        sum = c_m[0, 0] + c_m[1, 1] + c_m[2, 2] + c_m[3, 3] + c_m[4, 4] # micro-F1   micro-F1 equals to (sum of TP)/(sum of all labels), because the bottom is a constant, it is omitted
        print("epoch %d TP on train set : %d" % (epoch, sum))
        train_sum.append(sum)
        print(train_loss)
        print(train_sum)
        if opt.save_model:
            save_path = os.path.join(opt.model_save_path, f'epoch_{epoch}_final.pth')
            torch.save(model.module.state_dict(), save_path)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            c_m = torch.zeros(5, 5)
            dev_sum = []
            dev_loss = []
            for i, batch in enumerate(opt.test):

                attention_mask, input_ids, padded_labels, org_len, mfcc, source, id = batch   #size:batch, length
                attention_mask = attention_mask.cuda()
                input_ids = input_ids.cuda()
                padded_labels = padded_labels.cuda()
                mfcc = mfcc.cuda()
                mfcc_len = [mfcc[i].size(0) for i in range(0, mfcc.size(0))]
                src_mask = make_non_pad_mask(mfcc_len).unsqueeze(-2).cuda()

                preds, attention_map = model(mfcc=mfcc, input_ids=input_ids, attention_mask=attention_mask, mfcc_mask=None, src_mask=src_mask)

                loss_mask = attention_mask.bool()  # b, len
                temp_mask = torch.ones(attention_mask.size()).cuda() - attention_mask  # (B，length, 3)
                masked_pre = preds.masked_fill(temp_mask.unsqueeze(-1).bool(), value=torch.tensor(0))  # (B，length, 3)
                masked_pred_label = torch.argmax(torch.softmax(input=masked_pre, dim=-1), dim=-1)  # B, len
                loss = F.cross_entropy(masked_pre[loss_mask], padded_labels[loss_mask])  # b, 3   b
                test_loss += loss.item()

                padded_labels = padded_labels.int()
                masked_pred_label = masked_pred_label.int()
                for b in range(0, padded_labels.size(0)):
                    l = padded_labels[b]
                    p = masked_pred_label[b]
                    for id in range(0, len(l)):
                        label = l[id]
                        pred = p[id]
                        c_m[label, pred] += 1

            dev_loss.append(test_loss / (i + 1))
            print("epoch %d loss on dev  set : loss: %.3f " % (epoch, test_loss / (i + 1)))
            print("comfusion_matrix")
            print(c_m)
            sum = c_m[0, 0] + c_m[1, 1] + c_m[2, 2] + c_m[3, 3] + c_m[4, 4]
            print("epoch %d TP on dev set : TP %d" % (epoch, sum))
            dev_sum.append(sum)
            print(dev_loss)
            print(dev_sum)

    return dev_sum.index(max(dev_sum))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default='asr', type=str)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=20)
    parser.add_argument('-printevery', type=int, default=50)
    parser.add_argument('-saveevery', type=int, default=2000)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-max_mfcc_length', type=int, default=4213)
    parser.add_argument('-bert_embedding_length', type=int, default=512)
    parser.add_argument('-d_mfcc', type=int, default=256)
    parser.add_argument('-d_text', type=int, default=768)
    parser.add_argument('-device_ids', type=list, default=[i for i in range(0, 4)])
    parser.add_argument('-num_tags', type=int, default=5)
    parser.add_argument('-save_attention_map', type=bool, default=False)
    parser.add_argument('-save_model', type=bool, default=True)
    parser.add_argument('-load_pretrain_asr', type=bool, default=True)
    parser.add_argument('-asr_path', type=str,
                        default='./pretrained_asr')
    parser.add_argument('-model_save_path', type=str,
                        default='./model_save')
    parser.add_argument('-attention_map_save_path', type=str,
                        default='./attention_map')
    parser.add_argument('-bert_checkpoint', type=str,
                        default='./pretrained_bert')
    parser.add_argument('-pred_save_path', type=str,
                        default='./prediction_save')
    opt = parser.parse_args()


    opt.test_pred_save_path = os.path.join(opt.pred_save_path, 'test.txt')

    opt.gpu_ids = str(opt.device_ids)[1:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    train_text_path = './train.txt'
    dev_text_path = './dev.txt'


    scp_path = './feats.scp'
    loader = kaldiio.load_scp(scp_path)
    read_in = read_text(checkpoint=opt.bert_checkpoint, bert_embedding_length=opt.bert_embedding_length)
    train_dataset, dev_dataset = read_in.read_text_and_label(train_path=train_text_path, test_path=dev_text_path)

    train_data = Load_audio_and_text_data(dataset=train_dataset, bert_embedding_length=opt.bert_embedding_length, loader=loader)
    dev_data = Load_audio_and_text_data(dataset=dev_dataset, bert_embedding_length=opt.bert_embedding_length, loader=loader)

    opt.train = DataLoader(dataset=train_data, batch_size=opt.batchsize, shuffle=True, collate_fn=my_collate, num_workers=10,worker_init_fn=worker_init)
    opt.dev = DataLoader(dataset=dev_data, batch_size=opt.batchsize, shuffle=False, collate_fn=my_collate, num_workers=10,worker_init_fn=worker_init)

    model = get_model(opt)

    # scratch
    # params = list(model.named_parameters())
    # param_group = [
    #     {'params': [p for n, p in params if 'bert' in n], 'weight_decay': 1e-2, 'lr': 1e-5},
    #     {'params': [p for n, p in params if 'bert' not in n]}
    # ]


    # finetune
    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if ('bert' in n or 'sub_sampling' in n)], 'weight_decay': 1e-2, 'lr': 1e-5},
        {'params': [p for n, p in params if ('bert' not in n and 'sub_sampling' not in n)]}
    ]

    # fix
    # frozen_list = ['sub_sampling']
    # for name, param in model.named_parameters():
    #     if name.split('.')[0] in frozen_list:
    #         param.requires_grad = False
    # params = list(model.named_parameters())
    # param_group = [
    #     {'params': [p for n, p in params if ('bert' in n and p.requires_grad)], 'weight_decay': 1e-2, 'lr': 1e-5},
    #     {'params': [p for n, p in params if ('bert' not in n and p.requires_grad)]}
    # ]

    opt.optimizer = torch.optim.Adam(param_group, lr=opt.lr,
                                     betas=(0.9, 0.98), eps=1e-9)
    num_steps = len(opt.train) * opt.epochs
    opt.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.optimizer, T_max=num_steps)
    opt.warmup_scheduler = warmup.UntunedLinearWarmup(opt.optimizer)


    best_id = train_model(model, opt)

    predict_result(best_id, opt)

    calculate_pr(opt)

if __name__ == "__main__":
    main()

