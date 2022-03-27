import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 32, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.linear = nn.Linear(83, 40)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        :rtype: torch.Tensor
        """
        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)  #b, 1, 4213,  83

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.linear(xs_pad)) #b, 1, 4213,  40
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64
        ).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2) #b, 1000, 64, 10
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )
        return xs_pad, ilens  # no state in this layer


class VGGPreNet(nn.Module):
    """VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf"""

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): input dimension, e.g. number of Mel-freq banks.
        """
        super().__init__()
        self.hidden1_dim = 64
        self.hidden2_dim = 128
        in_channel, freq_dim, out_dim = self.get_vgg2l_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, self.hidden1_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden1_dim, self.hidden1_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.hidden1_dim, self.hidden2_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden2_dim, self.hidden2_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

    def get_vgg2l_dim(self, input_dim):
        """ Check input dimension, delta/delta-delta features should be stack over
         channels.
        Returns:
            tuple (input_channel, freq_dim, out_dim): out_dim is the output dimension
            of two-layer VGG.
        """
        if input_dim % 13 == 0:
            # MFCC features
            return int(input_dim / 13), 13, (13 // 4) * self.hidden2_dim
        elif input_dim % 80 == 0:
            # 80-dim Mel-spectrogram and its delta/delta-delta if any
            return int(input_dim / 80), 80, (80 // 4) * self.hidden2_dim
        else:
            raise ValueError(
                f"Currently only support input dimension 13/16/39 for MFCC or 80/160/240 for Mel-spec,but get {input_dim}.")

    def reshape_to_4D(self, x, x_len):
        # Downsample along time axis
        x_len = x_len // 4
        # Crop sequence such that the time-dim is divisable by 4
        if x.size(1) % 4 != 0:
            x = x[:, :-(x.size(1) % 4), :].contiguous()
        bs, ts, ds = x.size()
        x = x.view(bs, ts, self.in_channel, self.freq_dim)
        # (B, channels, T, freq_dim)
        x = x.transpose(1, 2)
        return x, x_len

    def forward(self, x, x_len):
        """
        Args:
            x (tensor): shape [B, T, D]
            x_len (tensor): shape [B, ]
        """
        x, x_len = self.reshape_to_4D(x, x_len)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        # (B, T//4, 32*D)
        x = x.contiguous().view(
            x.size(0), x.size(1), self.out_dim)
        return x, x_len


if __name__ == "__main__":
    # import pdb;
    #
    # pdb.set_trace()
    SubSampleCNN = VGG2L()
    xs_pad, ilens = torch.randn(5, 4213, 83), torch.Tensor([4213] * 5)
    y, ilens = SubSampleCNN(xs_pad, ilens)   #时间步维度变为原来四分之一
    print(y.shape)
    print(ilens)

    # xs_pad, ilens = torch.randn(5, 35, 80), torch.Tensor([35] * 5)
    # SubSampleCNN = VGGPreNet(80)
    # y, ilens = SubSampleCNN(xs_pad, ilens)
    # print(y.shape)
    # print(ilens)