import os
import errno
import typing as tp
import warnings

import torch
from torch import nn
import numpy as np

TORCHAUDIO_AVAILABLE: bool = True
try:
    import torchaudio
except BaseException as e:
    TORCHAUDIO_AVAILABLE = False
    warnings.warn("Torchaudio is not available!")

AUDIOAUG_AVAILABLE: bool = True
try:
    from audioaug import (FeatureParams, Resample, extract_features, read_waveform,
                          apply_augmentation, WindowType, PaddingType, PreemphasisInput,
                          AudioInfo)
except BaseException as e:
    AUDIOAUG_AVAILABLE: bool = False
    warnings.warn(f"Audioaug is not available! Error when importing module: {e}")

FEATURE_EXTRACTOR_REGISTRY = {}


class AudioaugFeatureExtractor:
    """Read waveforms, apply augmentations and extract features.
    """
    MIN_AUG_DURATION = 0.2

    def __init__(
        self,
        extraction_config: tp.Dict = None,
        sample_rate: int = 16000,
    ):
        self.perform_feature_extraction = extraction_config is not None
        self.wav_scaling = True
        if self.perform_feature_extraction:
            extraction_config["sampling_freq"] = sample_rate
            extraction_config["high_freq_filterbank"] = sample_rate // 2
            self.features_params = FeatureParams(**self.read_config(extraction_config))
            self.n_features = self.features_params.num_filterbank_chans
            self.sample_rate = self.features_params.sampling_freq
            self.wav_scaling = self.features_params.wav_scaling
        else:
            self.sample_rate = sample_rate

        self.default_info = AudioInfo(sample_rate=self.sample_rate, channels=1)

    def process(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

        read_dtype = np.float32 if self.wav_scaling else np.int16
        wav, info = read_waveform(filepath, dtype=read_dtype)
        duration = len(wav) / info.sample_rate

        if info.sample_rate != self.sample_rate:
            aug = Resample(self.sample_rate)
            wav, info = apply_augmentation(wav, info, [aug], return_dtype=wav.dtype)

        if self.perform_feature_extraction:
            features = extract_features(wav, info, self.features_params)
            features = torch.from_numpy(features)
            features = features.view(-1, self.n_features)
            return features
        else:
            return torch.from_numpy(wav)

    def extract_features(self, wav: np.ndarray):
        features = extract_features(wav, self.default_info, self.features_params)
        features = torch.from_numpy(features)
        features = features.view(-1, self.n_features)
        return features

    @staticmethod
    def read_config(extraction_config):
        def read_window_type(window_type):
            if window_type == "hamming":
                return WindowType.HAMMING
            return WindowType.HANNING

        def read_padding(padding):
            if padding == "reflect":
                return PaddingType.REFLECT
            return PaddingType.CONSTANT

        def read_preemph_input(preemph_input):
            if preemph_input == "audio":
                return PreemphasisInput.AUDIO
            return PreemphasisInput.FRAMES

        config = extraction_config
        config["window_type"] = read_window_type(config.get("window_type"))
        config["padding"] = read_padding(config.get("padding"))
        config["preemph_input"] = read_preemph_input(config.get("preemph_input"))
        return config


if AUDIOAUG_AVAILABLE:
    FEATURE_EXTRACTOR_REGISTRY['audioaug'] = AudioaugFeatureExtractor


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class TorchAudioFeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """
    def __init__(self, sample_rate: int, features: int):
        super().__init__()
        self.sample_rate = sample_rate

        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)

    def process(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

        wav, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        return self.featurizer(wav)[0].transpose(0, 1)


if TORCHAUDIO_AVAILABLE:
    FEATURE_EXTRACTOR_REGISTRY['torchaudio'] = TorchAudioFeatureExtractor
