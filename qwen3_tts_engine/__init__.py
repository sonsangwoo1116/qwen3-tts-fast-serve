"""Qwen3-TTS with vLLM-style optimizations."""

from qwen3_tts_engine.sampling_params import SamplingParams
from qwen3_tts_engine.config import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig

__version__ = "0.1.0"

__all__ = [
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "SamplingParams",
]
