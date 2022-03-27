import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import PositionalEncoding

from espnet_local.nets.pytorch_backend.conformer.encoder import Encoder
from espnet_local.nets.pytorch_backend.nets_utils import make_non_pad_mask


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    SubSampleCNN = Conv2dSubsampling(idim=83, odim=256, dropout_rate=0.1, pos_enc=None)

    # load pretrained model
    pretrained_model = torch.load("snapshot.iter.380000")['model']
    enc_dic = {}
    for key in SubSampleCNN.state_dict():
        enc_dic[key] = pretrained_model["encoder.embed." + key]
    # import pdb; pdb.set_trace()
    SubSampleCNN.load_state_dict(enc_dic)

    # Inference test

    x = torch.randn(5, 100, 83)
    # import pdb; pdb.set_trace()
    y, _ = SubSampleCNN(x, None)
    print(y.size())

    ### Conformer Encoder ###
    conformer_encoder = Encoder(
        idim=83,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=12,
        input_layer="conv2d",
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        pos_enc_layer_type="legacy_rel_pos",
        selfattention_layer_type="legacy_rel_selfattn",
        activation_type="swish",
        macaron_style=1,
        use_cnn_module=1,
        zero_triu=False,
        cnn_module_kernel=31,
    )
    enc_dic = {}
    # import pdb; pdb.set_trace()
    for key in conformer_encoder.state_dict():
        try:
            enc_dic[key] = pretrained_model["encoder." + key]
        except:
            print(key)
    conformer_encoder.load_state_dict(enc_dic)
    # import pdb; pdb.set_trace()

    # encoder.encoders.0.conv_module.norm.weight
    # encoders.0.conv_module.norm.weight

    x = torch.randn(5, 100, 83)
    ilens = [100, 95, 90, 90, 88]
    src_mask = make_non_pad_mask(ilens).unsqueeze(-2)

    # import pdb; pdb.set_trace()
    y, _ = conformer_encoder(x, src_mask)
    print(y.size())