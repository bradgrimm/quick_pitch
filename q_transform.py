import numpy as np
import torch
from torch import nn
from nnAudio.features import CQT

from constants import ANNOTATIONS_N_SEMITONES, MAX_N_SEMITONES, AUDIO_SAMPLE_RATE, FFT_HOP, ANNOTATIONS_BASE_FREQUENCY, \
    CONTOURS_BINS_PER_SEMITONE


class NormalizedCQT(nn.Module):
    def __init__(self, n_harmonics: int, use_batchnorm: bool):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        n_semitones = np.min([
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ])
        n_bins = n_semitones * CONTOURS_BINS_PER_SEMITONE
        self.cqt = CQT(
            sr=AUDIO_SAMPLE_RATE,
            hop_length=FFT_HOP,
            fmin=ANNOTATIONS_BASE_FREQUENCY,
            n_bins=n_bins,
            bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        )
        self.normalized_log = NormalizedLog()
        if use_batchnorm:
            self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, inputs):
        """Calculate the CQT of the input audio.

        Input shape: (batch, number of audio samples, 1)
        Output shape: (batch, number of frequency bins, number of time frames)

        Args:
            inputs: The audio input.
            n_harmonics: The number of harmonics to capture above the maximum output frequency.
                Used to calculate the number of semitones for the CQT.
            use_batchnorm: If True, applies batch normalization after computing the CQT

        Returns:
            The log-normalized CQT of the input audio.
        """
        x = self.cqt(inputs)
        x = self.normalized_log(x)
        x = x.unsqueeze(1)  # Add channel
        if self.use_batchnorm:
            x = self.batch_norm(x)
        return x.transpose(3, 2)


class NormalizedLog(nn.Module):
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """
    def forward(self, inputs):
        # convert magnitude to power
        bs = inputs.shape[0]
        power = torch.square(inputs)
        log_power = 10 * log_base_b(power + 1e-10, 10)

        log_power_min = log_power.view(bs, -1).min(axis=1)[0].reshape(-1, 1, 1)
        log_power_offset = log_power - log_power_min
        log_power_offset_max = log_power_offset.view(bs, -1).max(axis=1)[0].reshape(bs, 1, 1)
        log_power_normalized = torch.where(log_power_offset_max != 0, log_power_offset / log_power_offset_max, torch.zeros_like(log_power_offset))

        return log_power_normalized.reshape(inputs.shape)


def log_base_b(x, base: int):
    """
    Compute log_b(x)
    Args:
        x : input
        base : log base. E.g. for log10 base=10
    Returns:
        log_base(x)
    """
    numerator = torch.log(x)
    denominator = np.log(base)
    return numerator / denominator