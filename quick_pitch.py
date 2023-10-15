from typing import List
import onnx
from onnx2pytorch import ConvertModel

import torch
from torch import nn
from torch.nn import functional as F

from constants import CONTOURS_BINS_PER_SEMITONE, N_FREQ_BINS_CONTOURS
from q_transform import NormalizedCQT, log_base_b


class BasicPitch(nn.Module):
    def __init__(
            self,
            n_harmonics: int = 8,
            n_filters_contour: int = 32,
            n_filters_onsets: int = 32,
            n_filters_notes: int = 32,
    ):
        super().__init__()
        self.normalized_cqt = NormalizedCQT(n_harmonics, use_batchnorm=True)
        harmonics = [0.5] + list(range(1, n_harmonics)) if n_harmonics > 1 else [1]
        self.harmonic_stacking = HarmonicStacking(CONTOURS_BINS_PER_SEMITONE, harmonics, N_FREQ_BINS_CONTOURS)
        self.contour = BasicPitchContour(n_filters_contour)

    def forward(self, x):
        x = self.normalized_cqt(x)
        x = self.harmonic_stacking(x)
        x = self.contour(x)
        return {"contour": x}


class BasicPitchContour(nn.Module):
    def __init__(self, n_filters_contour: int = 32):
        super().__init__()
        # self.contour_conv_1 = nn.Conv2d(8, n_filters_contour, (5, 5), padding="same")
        # self.contour_batch_norm_1 = nn.BatchNorm2d(n_filters_contour)
        # self.contour_activation_1 = nn.ReLU()
        self.contour_conv_2 = nn.Conv2d(8, 8, (3, 3 * 13), padding="same")
        self.contour_batch_norm_2 = nn.BatchNorm2d(8)
        self.contour_activation_2 = nn.ReLU()
        self.contour_conv_3 = nn.Conv2d(8, 1, (5, 5), padding="same")
        self._load_weights_from_onnx()

    def _load_weights_from_onnx(self):
        onnx_model = onnx.load('/home/bgrimm/basic_pitch.onnx')
        pytorch_model = ConvertModel(onnx_model)
        m = list(pytorch_model.modules())
        # self.contour_conv_1.load_state_dict(m[-17].state_dict())
        self.contour_conv_2.load_state_dict(m[-15].state_dict())
        self.contour_conv_3.load_state_dict(m[-13].state_dict())

    def forward(self, x):
        # x = self.contour_conv_1(x)
        # x = self.contour_batch_norm_1(x)
        # x = F.relu(x)
        x = self.contour_conv_2(x)
        x = self.contour_batch_norm_2(x)
        x = F.relu(x)
        x = self.contour_conv_3(x)
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x


class HarmonicStacking(nn.Module):
    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """

    def __init__(self, bins_per_semitone: int, harmonics: List[float], n_output_freqs: int):
        """Downsample frequency by stride, upsample channels by 4."""
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        shifts = 12.0 * bins_per_semitone * log_base_b(torch.tensor(harmonics), 2)
        self.shifts = torch.round(shifts, decimals=2).int()
        self.n_output_freqs = n_output_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (n_batch, n_times, n_freqs, 1)
        assert len(x.shape) == 4
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                paddings = (0, shift)
                padded = F.pad(x[:, :, :, shift:], paddings)
            elif shift < 0:
                paddings = (-shift, 0)
                padded = F.pad(x[:, :, :, :shift], paddings)
            else:
                raise ValueError
            channels.append(padded)

        x = torch.cat(channels, dim=1)
        x = x[:, :, :, :self.n_output_freqs]  # return only the first n_output_freqs frequency channels
        return x


