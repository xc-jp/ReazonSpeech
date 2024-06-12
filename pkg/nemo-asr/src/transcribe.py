import os
from typing import Literal
from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch
from .interface import AudioData, TranscribeConfig, TranscribeResult
from .decode import decode_hypothesis, PAD_SECONDS
from .audio import audio_to_file, pad_audio, norm_audio
from .fs import create_tempfile


def load_model(
    device: Literal["cuda", "cpu"] | torch.device | None = None,
) -> EncDecRNNTBPEModel:
    """Load ReazonSpeech model

    Args:
      device: Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    from nemo.utils import logging

    logging.setLevel(logging.ERROR)

    return EncDecRNNTBPEModel.from_pretrained(
        "reazon-research/reazonspeech-nemo-v2",
        map_location=device,  # pyright: ignore[reportArgumentType]
    )


def transcribe(
    model: EncDecRNNTBPEModel, audio: AudioData, config: TranscribeConfig | None = None
) -> TranscribeResult:
    """Inference audio data using NeMo model

    Args:
        model: ReazonSpeech model
        audio: Audio data to transcribe
        config: Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = pad_audio(norm_audio(audio), PAD_SECONDS)

    # TODO Study NeMo's transcribe() function and make it
    # possible to pass waveforms on memory.
    with create_tempfile() as tmpf:
        audio_to_file(tmpf, audio)

        if os.name == "nt":
            tmpf.close()

        hyp, _ = model.transcribe(
            [tmpf.name], batch_size=1, return_hypotheses=True, verbose=config.verbose
        )
        hyp = hyp[0]

    ret = decode_hypothesis(model, hyp)

    if config.raw_hypothesis:
        ret.hypothesis = hyp

    return ret
