"""A small, fast network that stands in for Beat This!.

The pipeline only asks the network for a rough pulse, phase and bar — the millisecond work
is done on the audio — so the network can be far smaller than Beat This!, as long as it
gets those right. This is a temporal convolutional network in the style of Böck & Davies
(2019/2020): a convolutional front end that folds the 128 mel bands into a feature vector
per frame, then a stack of dilated 1-D convolutions that sees ~80 s either side, then two
outputs per frame (beat, downbeat logits). Same input as Beat This! (log-mel, 22.05 kHz,
hop 441 = 50 fps), so its teacher's outputs line up frame for frame.

Convolutions only: cheap on a CPU, trivially exported to ONNX, easy to port.
"""

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation * 2, padding=pad * 2)
        self.mix = nn.Conv1d(2 * channels, channels, 1)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Two dilation rates side by side, as in Böck & Davies 2020.
        y = torch.cat([self.act(self.conv1(x)), self.act(self.conv2(x))], dim=1)
        return x + self.drop(self.mix(y))


class Student(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 10, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks, "kernel": kernel, "dropout": dropout}
        # Strided in frequency from the first layer: the elementwise work on wide
        # (channels × bands × frames) tensors, not the convolutions, is what costs on a CPU.
        self.front = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), stride=(1, 3), padding=(1, 1)), nn.ELU(),   # 128 -> 43
            nn.Conv2d(16, 32, (3, 3), stride=(1, 3), padding=(1, 1)), nn.ELU(),  # 43 -> 15
            nn.Conv2d(32, channels, (1, 15)), nn.ELU(),                          # 15 -> 1
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(*[Block(channels, 2**i, kernel, dropout) for i in range(blocks)])
        self.head = nn.Conv1d(channels, 2, 1)

    def forward(self, spect: torch.Tensor) -> torch.Tensor:
        """(batch, frames, 128) log-mel -> (batch, frames, 2) beat/downbeat logits."""
        x = self.front(spect.unsqueeze(1))  # (B, C, T, 1)
        x = self.tcn(x.squeeze(-1))
        return self.head(x).transpose(1, 2)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
