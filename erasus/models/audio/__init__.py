"""
erasus.models.audio — Audio model wrappers.
"""

from erasus.models.audio.whisper import WhisperWrapper
from erasus.models.audio.wav2vec import Wav2VecWrapper
from erasus.models.audio.clap import CLAPWrapper

__all__ = ["WhisperWrapper", "Wav2VecWrapper", "CLAPWrapper"]
