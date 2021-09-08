import torch
from torch import nn
import torch.nn.functional as F


from .base_model import BaseModel
from .basic_module import (
    OnehotMSA,
    BasicBlock,
)


class Model(BaseModel):
    def __init__(
        self, emb_ninp, block1_channels, block1_repeats, block2_channels, block2_repeats
    ):
        super(Model, self).__init__()
        self.onehot = OnehotMSA(emb_ninp)
        self.inplanes = emb_ninp * 2
        self._blocks_1 = self._make_layers(block1_channels, block1_repeats, BasicBlock)
        self._blocks_2 = self._make_layers(block2_channels, block2_repeats, BasicBlock)
        self.fc = nn.Conv1d(self.inplanes, 15, kernel_size=1, bias=True)

    def _make_layers(self, channels, repeats, block):
        layers = []
        for c, r in zip(channels, repeats):
            for _ in range(r):
                layers.append(block(self.inplanes, c))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, data):
        seq = data["seq"]
        msa = data["msa"]

        B, K, L = msa.shape
        seq = self.onehot(seq)
        msa = self.onehot(msa)
        x = torch.cat([seq[:, None, :, :].repeat(1, msa.shape[1], 1, 1), msa], dim=2)
        x = x.reshape(-1, *x.shape[2:])
        x = self._blocks_1(x)
        x = x.reshape(-1, K, *x.shape[1:])

        x = torch.max(x, dim=1)[0]
        x = self._blocks_2(x)
        x = self.fc(x).permute(0, 2, 1).contiguous()
        angle = torch.tanh(x[:, :, 11:])
        return {
            "ss3": x[:, :, :3],
            "ss8": x[:, :, 3:11],
            "sin_phi": angle[:, :, 0],
            "cos_phi": angle[:, :, 1],
            "sin_psi": angle[:, :, 2],
            "cos_psi": angle[:, :, 3],
        }
