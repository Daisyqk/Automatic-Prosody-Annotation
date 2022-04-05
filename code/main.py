import argparse
import os
import torch
from Models import get_model
from load_data import read_text
from load_data import Load_audio_and_text_data
from torch.utils.data import DataLoader
import numpy as np
import kaldiio
from my_collate import my_collate
from predict_result import predict_result
from espnet_local.nets.pytorch_backend.nets_utils import make_non_pad_mask


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_mfcc_length', type=int, default=4213)
    parser.add_argument('-bert_embedding_length', type=int, default=512)
    parser.add_argument('-d_mfcc', type=int, default=256)
    parser.add_argument('-d_text', type=int, default=768)
    parser.add_argument('-device_ids', type=list, default=[i for i in range(0, 1)])
    parser.add_argument('-num_tags', type=int, default=5)
    parser.add_argument('-save_attention_map', type=bool, default=False)
    parser.add_argument('-save_pred_res', type=bool, default=True)
    parser.add_argument('-attention_map_save_path', type=str,
                        default='./../attention_map')
    parser.add_argument('-bert_checkpoint', type=str,
                        default='./../bert/vocab.txt')
    parser.add_argument('-pred_save_path', type=str,
                        default='./../prediction_save')
    opt = parser.parse_args()


    opt.test_pred_save_path = os.path.join(opt.pred_save_path, 'test.txt')

    opt.gpu_ids = str(opt.device_ids)[1:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    opt.test_text_path = './../data/test.txt'
    opt.scp_path = './../data/feature.scp'
    opt.model_path = './../model.pth'

    predict_result(opt)


if __name__ == "__main__":
    main()

