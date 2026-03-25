"""
TTS Server using qwen3_tts_engine for Qwen3-TTS inference.

OpenAI-compatible API:
  POST /v1/audio/speech       - TTS with preset/custom voice (streaming + non-streaming)
  POST /v1/audio/voice-clone  - Voice cloning with inline reference audio
  GET  /v1/voices             - List available voices
  GET  /health                - Health check

Architecture:
  Single BatchedStreamingEngine: one model instance (~7GB VRAM) serves N
  concurrent requests via batched Talker/Predictor step() calls.
  Streaming mode: codec chunks decoded to PCM and streamed via chunked HTTP.
"""

import asyncio
import base64
import io
import logging
import os
import struct
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts-server")

# ── Config ───────────────────────────────────────────────────
MODEL_ID = os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
CUSTOM_VOICES_DIR = Path(os.getenv("TTS_CUSTOM_VOICES", "/app/custom_voices"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8880"))
WARMUP = os.getenv("TTS_WARMUP_ON_START", "false").lower() in ("true", "1", "yes")
MAX_CONCURRENT = int(os.getenv("TTS_MAX_CONCURRENT", "64"))
STREAM_CHUNK_FRAMES = int(os.getenv("TTS_STREAM_CHUNK_FRAMES", "10"))
SAMPLE_RATE = 24000
# Built-in speakers (no voice clone needed)
BUILTIN_SPEAKERS = {"Vivian", "Alisa", "Layla"}  # Known Qwen3-TTS speakers

# ── Global state ─────────────────────────────────────────────
voice_prompts: dict = {}  # name -> voice_clone_prompt dict
tts_interface = None       # Qwen3TTSInterface (single instance)
batched_engine = None      # BatchedStreamingEngine
_prompt_lock = threading.Lock()

app = FastAPI(title="Qwen3-TTS nano-vllm", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──────────────────────────────────
class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "default"
    language: Optional[str] = "Auto"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False
    raw: Optional[bool] = False  # debug: skip all post-processing


class VoiceCloneRequest(BaseModel):
    input: str
    ref_audio: str  # base64
    ref_text: Optional[str] = ""
    x_vector_only_mode: Optional[bool] = False
    language: Optional[str] = "Auto"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False


# ── Helpers ──────────────────────────────────────────────────
def _normalize_volume(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to target peak level."""
    if len(audio) == 0:
        return audio
    if audio.dtype == np.int16:
        peak = np.max(np.abs(audio)).astype(np.float32)
        if peak < 1:
            return audio
        target = target_peak * 32767
        gain = target / peak
        return np.clip(audio.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
    else:
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            return audio
        gain = target_peak / peak
        return audio * gain


# Fixed gain for model output (model outputs ~30% of full scale)
_OUTPUT_GAIN = 2.5


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _pcm_to_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """WAV header with unknown data size (0xFFFFFFFF) for streaming."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', (36 + data_size) & 0xFFFFFFFF, b'WAVE',
        b'fmt ', 16, 1,
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b'data', data_size,
    )


# Causal decoder constants
_LEFT_CONTEXT = 25       # frames of left context for causal decode
_SAMPLES_PER_FRAME = 1920  # decoder total_upsample (12Hz @ 24kHz)
_START_PAD = 0           # skip first N codec frames (0 = disabled, not needed with good ref audio)
_END_PAD = 3             # padding frames at end to avoid causal conv right-edge artifact
_FADE_OUT_MS = 10        # fade-out duration at stream end (ms)


@torch.inference_mode()
def _decode_full(all_codes: list, ref_code=None) -> np.ndarray:
    """Decode ALL accumulated codes using the decoder's chunked_decode.

    Uses the native causal chunked_decode (chunk_size=300, left_context=25)
    which handles chunk boundaries properly. Quality matches single-request.

    If ref_code is provided, prepend it to give the decoder acoustic context
    (prevents cold-start onset artifacts), then trim the reference portion.
    """
    if not all_codes:
        return np.array([], dtype=np.int16)
    codes_tensor = torch.tensor(all_codes, dtype=torch.long, device="cuda")
    # Clamp negative codes to 0 (matches official speech_tokenizer.decode path)
    codes_tensor = torch.clamp(codes_tensor, min=0)

    # Prepend reference codec tokens for decoder context
    ref_len = 0
    if ref_code is not None:
        ref_code_t = ref_code.to(codes_tensor.device)
        # ref_code is [time, 16], same shape as codes_tensor
        ref_len = ref_code_t.shape[0]
        codes_tensor = torch.cat([ref_code_t, codes_tensor], dim=0)

    # [time, 16] → [1, 16, time]
    codes_formatted = codes_tensor.transpose(0, 1).unsqueeze(0)

    decoder = tts_interface.speech_tokenizer.tokenizer.model.decoder
    wav = decoder.chunked_decode(codes_formatted, chunk_size=300, left_context_size=25)

    audio = wav.squeeze().clamp(-1, 1).cpu().float().numpy()

    # Trim reference portion (proportional cut)
    if ref_len > 0:
        total_frames = codes_tensor.shape[0]
        cut_samples = int(ref_len / total_frames * len(audio))
        audio = audio[cut_samples:]

    # Skip first _START_PAD frames of decoded audio (model onset artifact)
    if _START_PAD > 0:
        skip = _START_PAD * 1920
        audio = audio[skip:]

    pcm = np.clip(audio * 32767 * _OUTPUT_GAIN, -32767, 32767).astype(np.int16)
    return _onset_limiter(pcm)


def _onset_limiter(pcm: np.ndarray) -> np.ndarray:
    """Normalize first 300ms RMS to a target range with single gain + crossfade.

    Uses one gain value for the entire onset region to avoid block-boundary
    artifacts. RMS-based is more perceptually consistent than peak-based.
    """
    if len(pcm) == 0:
        return pcm

    _TARGET_RMS = 4500.0
    _MIN_RMS = 3000.0
    _MAX_RMS = 6000.0
    _ONSET_MS = 300
    onset_samples = min(len(pcm), int(SAMPLE_RATE * _ONSET_MS / 1000))
    if onset_samples < SAMPLE_RATE // 10:  # < 100ms, skip
        return pcm

    onset = pcm[:onset_samples].astype(np.float32)
    rms = np.sqrt(np.mean(onset ** 2))

    if rms < 500:  # silence
        return pcm
    if _MIN_RMS <= rms <= _MAX_RMS:  # already good
        return pcm

    gain = _TARGET_RMS / rms
    # Clamp gain to avoid extreme amplification/reduction
    gain = max(0.4, min(gain, 2.5))

    onset_gained = onset * gain

    # Crossfade last 50ms into original to avoid discontinuity
    xfade_len = min(int(SAMPLE_RATE * 0.05), onset_samples // 4)
    if onset_samples < len(pcm) and xfade_len > 0:
        xfade = np.linspace(1.0, 0.0, xfade_len, dtype=np.float32)
        tail_start = onset_samples - xfade_len
        original_tail = pcm[tail_start:onset_samples].astype(np.float32)
        onset_gained[tail_start:] = (
            onset_gained[tail_start:] * xfade
            + original_tail * (1.0 - xfade)
        )

    pcm[:onset_samples] = np.clip(onset_gained, -32767, 32767).astype(np.int16)
    return pcm


@torch.inference_mode()
def _decode_incremental(codes_slice: list) -> np.ndarray:
    """Decode a slice of codes (context + new frames) incrementally.

    Much cheaper than _decode_full when only new frames are needed.
    The caller provides left_context frames before the new frames.
    """
    if not codes_slice:
        return np.array([], dtype=np.int16)
    codes_tensor = torch.tensor(codes_slice, dtype=torch.long, device="cuda")
    codes_tensor = torch.clamp(codes_tensor, min=0)
    codes_formatted = codes_tensor.transpose(0, 1).unsqueeze(0)

    decoder = tts_interface.speech_tokenizer.tokenizer.model.decoder
    wav = decoder.chunked_decode(codes_formatted, chunk_size=300, left_context_size=25)

    audio = wav.squeeze().clamp(-1, 1).cpu().float().numpy()
    return np.clip(audio * 32767 * _OUTPUT_GAIN, -32767, 32767).astype(np.int16)


async def _streaming_pcm(chunk_q, chunk_frames: int, ref_code=None):
    """Streaming PCM generator using full re-decode (chunked_decode).

    Each decode call re-decodes ALL accumulated codes via chunked_decode,
    then outputs only the new portion. Includes simple AGC to keep volume
    consistent across the utterance.
    """
    yield _pcm_to_wav_header(SAMPLE_RATE)
    gate = _LeadSilenceGate()
    all_codes = []
    prev_samples = 0  # PCM samples already sent

    # Upward-only AGC: boost quiet chunks to target RMS, leave loud chunks alone
    _AGC_TARGET = 2000.0
    _AGC_MAX_GAIN = 5.0

    def _apply_agc(pcm_arr: np.ndarray) -> np.ndarray:
        if len(pcm_arr) == 0:
            return pcm_arr
        rms = np.sqrt(np.mean(pcm_arr.astype(np.float32) ** 2))
        if rms < 100 or rms >= _AGC_TARGET:  # silence or already loud enough
            return pcm_arr
        gain = min(_AGC_MAX_GAIN, _AGC_TARGET / rms)
        return np.clip(pcm_arr.astype(np.float32) * gain, -32767, 32767).astype(np.int16)

    while True:
        chunk = await chunk_q.get()
        if chunk is None:
            break
        all_codes.append(chunk)
        if len(all_codes) % chunk_frames == 0:
            pcm_arr = _decode_full(all_codes, ref_code=ref_code)
            if len(pcm_arr) > prev_samples:
                new_pcm = pcm_arr[prev_samples:]
                prev_samples = len(pcm_arr)
                new_pcm = _apply_agc(new_pcm)
                out = gate.process(new_pcm.tobytes())
                if out:
                    yield out

    # Flush remaining
    if all_codes:
        pcm_arr = _decode_full(all_codes, ref_code=ref_code)
        if len(pcm_arr) > prev_samples:
            new_pcm = pcm_arr[prev_samples:]
            new_pcm = _apply_agc(new_pcm)
            out = gate.process(new_pcm.tobytes())
            if out:
                yield out



# ── Silence handling ─────────────────────────────────────────
# Leading silence threshold: model outputs low-amp noise before real speech
# Real speech starts at ~1000+
_LEAD_THRESH = 1000
# Internal silence threshold: lower to avoid cutting quiet speech
_INTERNAL_THRESH = 300
# Max allowed internal silence: 500ms at 24kHz
_MAX_INTERNAL_SILENCE = int(SAMPLE_RATE * 0.5)  # 12000 samples
# Leading trim margin: keep 20ms before first voice
_LEAD_MARGIN = int(SAMPLE_RATE * 0.02)  # 480 samples


def _trim_wav_silence(wav_bytes: bytes) -> bytes:
    """Trim leading silence/artifacts and compress long internal silence gaps in WAV.

    The decoder produces a false burst (100-300ms) before real speech.
    We detect this by finding the first sustained voice segment (>200ms)
    rather than just the first sample above threshold.
    """
    if len(wav_bytes) <= 44:
        return wav_bytes
    arr = np.frombuffer(wav_bytes[44:], dtype=np.int16).copy()
    if len(arr) == 0:
        return wav_bytes

    # 1) Find real speech start: first sustained voice segment (>200ms)
    #    Skip initial decoder burst artifacts by requiring sustained energy
    block_ms = 20
    block_size = int(SAMPLE_RATE * block_ms / 1000)  # 480 samples per block
    min_sustained_blocks = 6  # 120ms of sustained voice = real speech

    voice_blocks = []
    for i in range(0, len(arr), block_size):
        chunk = arr[i:i + block_size]
        peak = np.max(np.abs(chunk))
        voice_blocks.append(peak > _LEAD_THRESH)

    # Find first run of min_sustained_blocks consecutive voice blocks
    run = 0
    real_start_block = 0
    for i, is_voice in enumerate(voice_blocks):
        if is_voice:
            run += 1
            if run >= min_sustained_blocks:
                real_start_block = i - min_sustained_blocks + 1
                break
        else:
            run = 0

    if run < min_sustained_blocks:
        # Fallback: use simple threshold
        voice = np.abs(arr) > _LEAD_THRESH
        if not voice.any():
            return wav_bytes
        first = max(0, np.argmax(voice) - _LEAD_MARGIN)
        arr = arr[first:]
    else:
        first = real_start_block * block_size
        arr = arr[first:]

    # 2) Compress internal silence (use lower threshold to preserve quiet speech)
    out = []
    silence_run = 0
    block = int(SAMPLE_RATE * 0.05)  # 50ms blocks
    for i in range(0, len(arr), block):
        chunk = arr[i:i + block]
        if np.max(np.abs(chunk)) <= _INTERNAL_THRESH:
            silence_run += len(chunk)
            if silence_run <= _MAX_INTERNAL_SILENCE:
                out.append(chunk)
        else:
            silence_run = 0
            out.append(chunk)

    if not out:
        return wav_bytes
    trimmed = np.concatenate(out)

    # 3) Fade-in 10ms
    fade_in_samples = int(SAMPLE_RATE * 0.01)
    if len(trimmed) > fade_in_samples:
        fade_curve = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        trimmed[:fade_in_samples] = (trimmed[:fade_in_samples].astype(np.float32) * fade_curve).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, trimmed.astype(np.float32) / 32767.0, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


class _LeadSilenceGate:
    """Streaming: skip leading artifacts and silence, compress internal silence.

    The decoder produces a false burst (100-400ms) before real speech,
    followed by a silence gap, then the actual speech. We buffer all
    initial audio and analyze it at 20ms block granularity to find the
    first sustained voice segment (>200ms without silence gap >80ms).
    """

    _BLOCK_MS = 20
    _BLOCK_SIZE = int(SAMPLE_RATE * _BLOCK_MS / 1000)  # 480 samples
    _SUSTAINED_BLOCKS = 6  # 120ms of sustained voice = real speech
    _SILENCE_GAP_BLOCKS = 4  # 80ms silence gap resets burst detection

    def __init__(self):
        self.started = False
        self.silence_run = 0
        self._raw_buffer = []  # raw PCM samples buffered before speech confirmed

    def process(self, pcm: bytes) -> bytes:
        if len(pcm) < 2:
            return b''
        arr = np.frombuffer(pcm, dtype=np.int16)

        if not self.started:
            self._raw_buffer.append(arr.copy())
            combined = np.concatenate(self._raw_buffer)

            # Analyze at 20ms block granularity
            blocks = []
            for i in range(0, len(combined), self._BLOCK_SIZE):
                chunk = combined[i:i + self._BLOCK_SIZE]
                if len(chunk) < self._BLOCK_SIZE // 2:
                    break
                blocks.append(np.max(np.abs(chunk)) > _LEAD_THRESH)

            # Find first run of _SUSTAINED_BLOCKS consecutive voice blocks
            # A silence gap of _SILENCE_GAP_BLOCKS resets the run counter
            run = 0
            silence_gap = 0
            real_start_block = None
            for i, is_voice in enumerate(blocks):
                if is_voice:
                    silence_gap = 0
                    run += 1
                    if run >= self._SUSTAINED_BLOCKS:
                        real_start_block = i - self._SUSTAINED_BLOCKS + 1
                        break
                else:
                    silence_gap += 1
                    if silence_gap >= self._SILENCE_GAP_BLOCKS:
                        run = 0  # reset: this was a burst, not real speech

            if real_start_block is not None:
                self.started = True
                self.silence_run = 0
                # Start from the detected real speech block
                start_sample = real_start_block * self._BLOCK_SIZE
                out = combined[start_sample:].copy()
                # Fade-in 10ms
                fade_in = int(SAMPLE_RATE * 0.01)
                if len(out) > fade_in:
                    fade_curve = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
                    out[:fade_in] = (out[:fade_in].astype(np.float32) * fade_curve).astype(np.int16)
                self._raw_buffer.clear()
                return out.tobytes()
            return b''

        # Audio started - use lower threshold for internal silence
        if np.max(np.abs(arr)) <= _INTERNAL_THRESH:
            self.silence_run += len(arr)
            if self.silence_run <= _MAX_INTERNAL_SILENCE:
                return pcm
            return b''
        else:
            self.silence_run = 0
            return pcm


def _find_voice(voice_name: str) -> Optional[dict]:
    prompt = voice_prompts.get(voice_name)
    if prompt is not None:
        return prompt
    for k, v in voice_prompts.items():
        if k.lower() == voice_name.lower():
            return v
    return None


# ── Custom voice loader ──────────────────────────────────────
def _load_custom_voices(iface):
    global voice_prompts
    if not CUSTOM_VOICES_DIR.exists():
        logger.info(f"Custom voices dir not found: {CUSTOM_VOICES_DIR}")
        return
    for voice_dir in sorted(CUSTOM_VOICES_DIR.iterdir()):
        if not voice_dir.is_dir() or voice_dir.name.startswith("."):
            continue
        ref_audio_path = voice_dir / "reference.wav"
        ref_text_path = voice_dir / "reference.txt"
        if not ref_audio_path.exists():
            logger.warning(f"No reference.wav in {voice_dir}, skipping")
            continue
        ref_text = ""
        if ref_text_path.exists():
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        try:
            t0 = time.time()
            prompt = iface.create_voice_clone_prompt(
                ref_audio=str(ref_audio_path),
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=not bool(ref_text),
            )
            voice_prompts[voice_dir.name] = prompt
            logger.info(f"Loaded voice '{voice_dir.name}' in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load voice {voice_dir.name}: {e}")


# ── Lifecycle ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global tts_interface, batched_engine
    from qwen3_tts_engine.interface import Qwen3TTSInterface

    logger.info(
        f"Loading single engine (max_concurrent={MAX_CONCURRENT})"
    )

    t0 = time.time()
    iface = Qwen3TTSInterface.from_pretrained(
        MODEL_ID,
        enforce_eager=False,
        tensor_parallel_size=1,
        talker_gpu_mem=0.5,
        predictor_gpu_mem=0.7,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    _load_custom_voices(iface)
    logger.info(f"Custom voices: {list(voice_prompts.keys())}")

    tts_interface = iface
    batched_engine = iface.create_batched_engine(max_concurrent=MAX_CONCURRENT)
    batched_engine.start()

    if WARMUP and voice_prompts:
        logger.info("Warming up engine...")
        loop = asyncio.get_event_loop()
        first_prompt = next(iter(voice_prompts.values()))
        try:
            wav = await batched_engine.submit("Hello.", "English", first_prompt, loop)
            logger.info(f"Warmup done ({len(wav)} bytes)")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")


# ── Endpoints ────────────────────────────────────────────────
@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    if batched_engine is None:
        raise HTTPException(503, "Model not loaded yet")

    # Check if it's a built-in speaker or a custom voice clone
    is_builtin = request.voice in BUILTIN_SPEAKERS
    prompt = None
    if not is_builtin:
        prompt = _find_voice(request.voice)
        if prompt is None:
            available = list(voice_prompts.keys()) + list(BUILTIN_SPEAKERS)
            raise HTTPException(
                400, f"Voice '{request.voice}' not found. Available: {available}"
            )

    loop = asyncio.get_event_loop()

    if request.stream:
        if is_builtin:
            chunk_q = batched_engine.submit_streaming_custom(
                request.input, request.language or "Korean", request.voice, loop,
            )
        else:
            chunk_q = batched_engine.submit_streaming(
                request.input, request.language or "Auto", prompt, loop,
            )

        # Extract ref_code for decoder context (ICL mode only)
        stream_ref_code = None
        if prompt and prompt.get("icl_mode") and prompt.get("ref_code") is not None:
            stream_ref_code = prompt["ref_code"]

        return StreamingResponse(
            _streaming_pcm(chunk_q, STREAM_CHUNK_FRAMES, ref_code=stream_ref_code),
            media_type="audio/wav",
            headers={
                "X-Stream": "true",
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )
    else:
        try:
            t0 = time.time()
            if is_builtin:
                wav_bytes = await batched_engine.submit_custom(
                    request.input, request.language or "Korean", request.voice, loop,
                )
            else:
                wav_bytes = await batched_engine.submit(
                    request.input, request.language or "Auto", prompt, loop,
                )
            if not request.raw:
                wav_bytes = _trim_wav_silence(wav_bytes)
                # Apply same upward AGC as streaming
                if len(wav_bytes) > 44:
                    arr = np.frombuffer(wav_bytes[44:], dtype=np.int16).copy()
                    if len(arr) > 0:
                        _BLOCK = SAMPLE_RATE // 2  # 0.5s blocks
                        _TARGET = 2000.0
                        _MAX_G = 5.0
                        for i in range(0, len(arr), _BLOCK):
                            blk = arr[i:i+_BLOCK]
                            rms = np.sqrt(np.mean(blk.astype(np.float32) ** 2))
                            if rms < 100 or rms >= _TARGET:
                                continue
                            g = min(_MAX_G, _TARGET / rms)
                            arr[i:i+_BLOCK] = np.clip(blk.astype(np.float32) * g, -32767, 32767).astype(np.int16)
                        buf = io.BytesIO()
                        sf.write(buf, arr.astype(np.float32) / 32767.0, SAMPLE_RATE, format="WAV")
                        wav_bytes = buf.getvalue()
            logger.info(
                f"[TTS] speech: {time.time()-t0:.2f}s, "
                f"voice={request.voice}, chars={len(request.input)}"
            )
            return Response(content=wav_bytes, media_type="audio/wav")
        except Exception as e:
            import traceback
            logger.error(f"TTS failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(500, str(e))


@app.post("/v1/audio/voice-clone")
async def voice_clone(request: VoiceCloneRequest):
    if batched_engine is None:
        raise HTTPException(503, "Model not loaded yet")

    t0 = time.time()
    try:
        audio_bytes = base64.b64decode(request.ref_audio)
        buf = io.BytesIO(audio_bytes)
        ref_audio, ref_sr = sf.read(buf)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1)

        with _prompt_lock:
            prompt = tts_interface.create_voice_clone_prompt(
                ref_audio=(ref_audio.astype(np.float32), ref_sr),
                ref_text=(
                    request.ref_text
                    if request.ref_text and not request.x_vector_only_mode
                    else None
                ),
                x_vector_only_mode=request.x_vector_only_mode,
            )

        loop = asyncio.get_event_loop()

        if request.stream:
            chunk_q = batched_engine.submit_streaming(
                request.input, request.language or "Auto", prompt, loop,
            )

            return StreamingResponse(
                _streaming_pcm(chunk_q, STREAM_CHUNK_FRAMES),
                media_type="audio/wav",
                headers={
                    "X-Stream": "true",
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Bit-Depth": "16",
                },
            )
        else:
            wav_bytes = await batched_engine.submit(
                request.input, request.language or "Auto", prompt, loop,
            )
            wav_bytes = _trim_wav_silence(wav_bytes)
            logger.info(
                f"[TTS] voice-clone: {time.time()-t0:.2f}s, "
                f"chars={len(request.input)}"
            )
            return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Voice clone failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/v1/voices")
async def list_voices():
    builtin = [
        {"id": n, "name": n, "type": "builtin", "description": f"Built-in speaker: {n}"}
        for n in sorted(BUILTIN_SPEAKERS)
    ]
    custom = [
        {"id": n, "name": n, "type": "custom", "description": f"Custom voice clone: {n}"}
        for n in sorted(voice_prompts.keys())
    ]
    return {"voices": builtin + custom}


@app.get("/health")
async def health():
    ready = batched_engine is not None
    return {
        "status": "healthy" if ready else "loading",
        "backend": {
            "name": "qwen3_tts_engine",
            "model_id": MODEL_ID,
            "ready": ready,
            "max_concurrent": MAX_CONCURRENT,
            "streaming": True,
            "stream_chunk_frames": STREAM_CHUNK_FRAMES,
        },
        "voices": list(voice_prompts.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
