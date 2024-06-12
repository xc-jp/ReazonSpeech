from __future__ import annotations

from typing import IO, TYPE_CHECKING

import librosa
import numpy as np
import numpy.typing as npt
import soundfile

from .interface import AudioData

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

SAMPLERATE = 16000


def audio_from_numpy(array: npt.NDArray[np.float32], samplerate: int) -> AudioData:
    """Load audio from Numpy array.

    Args:
      array: Audio audio
      samplerate: Sample rate of the input array
    """
    return AudioData(array, samplerate)


def audio_from_tensor(tensor: Tensor, samplerate: int) -> AudioData:
    """Load audio from PyTorch Tensor.

    Args:
      tensor: Audio audio
      samplerate: Sample rate of the input tensor
    """
    return audio_from_numpy(tensor.numpy(), samplerate)


def audio_from_path(path: str | Path) -> AudioData:
    """Load audio from a file.

    Args:
      path: Path to audio file
    """
    array, samplerate = librosa.load(path, sr=None)
    return audio_from_numpy(array, int(samplerate))


def audio_to_file(fp: IO, audio: AudioData, format: str | None = "wav") -> None:
    """Write audio data to file.

    Args:
      fp: output file
      audio: Audio data to write
      format: Audio encoding
    """
    soundfile.write(fp, audio.waveform, audio.samplerate, format=format)


def norm_audio(audio: AudioData) -> AudioData:
    """Normalize audio into 16khz mono waveform.

    Args:
      audio (AudioData): Audio data to normalize

    Returns:
      AudioData (16khz mono waveform)
    """
    waveform = audio.waveform
    if audio.samplerate != SAMPLERATE:
        waveform = librosa.resample(waveform, orig_sr=audio.samplerate, target_sr=SAMPLERATE)
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform)
    return AudioData(waveform, SAMPLERATE)


def pad_audio(audio: AudioData, seconds: float) -> AudioData:
    """Pad audio with silence.

    Args:
      audio: Audio data to pad
      seconds: Add N seconds padding

    Returns:
      AudioData
    """
    waveform = np.pad(audio.waveform, pad_width=int(seconds * audio.samplerate), mode="constant")
    return AudioData(waveform, audio.samplerate)
