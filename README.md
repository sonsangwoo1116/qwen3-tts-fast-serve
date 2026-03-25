# qwen3-tts-fast-serve

Optimized Qwen3-TTS serving engine with GPU-level performance optimizations and audio post-processing pipeline.

Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT) and [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm).

## Key Optimizations

### GPU Performance
- **Pre-allocated decode buffers** — CPU pinned memory + GPU buffers at init, `copy_(non_blocking=True)` async transfer. No per-step `torch.tensor()` + `.cuda()`
- **Block table caching** — Skip rebuild when batch size unchanged between decode steps
- **RoPE cos/sin pre-computation** — Full table computed at init for `max_position_embeddings`. Forward pass is pure index lookup (no matmul)
- **Batched embedding & logits** — When all sequences share the same generation step: single GPU kernel / single GEMM (fast path)
- **Finer CUDA Graph batch sizes** — `[1,2,4,6,8,10,12,14,16,20,24,28,32,48,...]` instead of `[1,2,4,8,16,32,48,...]`. Less padding waste for small batches
- **Predictor `step_all()`** — Drain all predictor decode steps in tight inner loop, reducing Python-level round-trip overhead
- **INT8 quantization** — Optional weight-only INT8 (`TTS_INT8_QUANTIZE=true`), better quality than INT4

### Audio Post-Processing
- **Temperature tuning** — Talker 0.9 → 0.8, Predictor 0.9 → 0.2. Dramatically more stable codec prediction
- **Ref-code prepend** — In ICL (voice clone) mode, prepend reference codec tokens to decoder input. Provides acoustic context, eliminates cold-start onset artifacts
- **Output gain (2.5x)** — Model output is only ~30% of full scale. Fixed gain correction applied after decode
- **Onset limiter** — First 300ms RMS normalized to target 4500 (gain range 0.4~2.5x, 50ms crossfade to original). Tames unstable initial decoder frames
- **Upward-only AGC** — Target RMS 2000, max 5x boost. Only amplifies quiet chunks; loud chunks pass through unchanged
- **Lead silence gate (streaming)** — 20ms block-level analysis. Requires 120ms continuous voiced audio before opening gate. 80ms+ silence gap resets burst detector. 10ms fade-in prevents clicks
- **Silence trim (non-streaming)** — Same sustained-voice algorithm. Skips decoder's 100-300ms fake bursts. 10ms fade-in on output

### Reference Audio Preprocessing
- Leading/trailing silence trim (threshold=0.01, 10ms margin)
- Peak normalization → 28000/32768 across all references
- 0.5s trailing silence padding (prevents phoneme bleed at end)
- UTMOS improvement: 3.864 → 4.014 (+3.9%) across 22 voices

## Benchmark Results

All benchmarks on **single NVIDIA RTX 3090 (24GB)**.

RTF / streaming / concurrency benchmarks use `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
UTMOS and TN accuracy evaluated on both 0.6B and 1.7B models.
All test sentences are in **Korean** (~20–60 chars per sentence).

### Per-Request RTF (Non-Streaming)

Each request generates complete audio before returning. RTF = generation time / audio duration.
RTF < 1.0 = faster than real-time.

| Concurrency | RTF avg | RTF max | Status |
|-------------|---------|---------|--------|
| 1 | 0.536 | — | Real-time |
| 2 | 0.526 | 0.536 | Real-time |
| 4 | 0.623 | 0.688 | Real-time |
| 8 | 0.741 | 0.820 | Real-time |
| 10 | 0.827 | 0.876 | Real-time |
| 16 | 1.078 | 1.352 | Near real-time |
| 20 | 1.226 | 1.347 | — |

**RTF < 1.0 up to 10 concurrent requests** on a single RTX 3090.

### Streaming Throughput

Audio streams back in chunks as they are generated (chunked PCM over HTTP).
Throughput RTF = total wall time / total audio duration across all channels.
TTFB = time to first audio chunk.

| Channels | Text Length | Success | Wall Time | Throughput RTF | Avg TTFB | P99 TTFB |
|----------|------------|---------|-----------|----------------|----------|----------|
| 20 | ~50 chars | 20/20 | 10.84s | 0.119 | 148ms | — |
| 30 | ~50 chars | 30/30 | 14.05s | 0.102 | 320ms | — |
| 40 | ~20 chars | 40/40 | 9.86s | 0.079 | 656ms | — |
| 40 | ~50 chars | 40/40 | 23.38s | 0.086 | 746ms | 829ms |

### Audio Quality (UTMOS)

UTMOS scale: 1.0–5.0 (correlates 0.93–0.96 with human MOS).

| Category | Avg UTMOS | Range |
|----------|-----------|-------|
| Male voices (10) | 3.9–4.3 | Consistent |
| Female voices (10) | 3.1–4.3 | Higher variance |
| **Overall (22 voices)** | **4.014** | — |

Quality criteria: Reference UTMOS >= 3.5, Generated avg >= 3.8, Generated min >= 3.0.

### Text Normalization Accuracy

Evaluated with TTS → ASR (Zipformer) → Levenshtein similarity. 22 voices x 20 test cases = 440 evaluations per model.

| Model | Overall | Phone Numbers | Voice Consistency |
|-------|---------|---------------|-------------------|
| 1.7B-Base | 97.7% | 93.5% | 94.5%–100.0% (5.5%p range) |
| 0.6B-Base | 93.2% | 77.2% | 80.3%–97.4% (17.1%p range) |

## Quick Start

```bash
git clone https://github.com/sonsangwoo1116/qwen3-tts-fast-serve.git
cd qwen3-tts-fast-serve
pip install -e .
```

### Docker

```bash
docker build -t qwen3-tts .
docker run --gpus '"device=0"' -p 8880:8880 \
  -e TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  qwen3-tts
```

### Run Server

```bash
export QWEN3_TTS_MODEL_PATH=/path/to/model
python -m uvicorn server:app --host 0.0.0.0 --port 8880
```

### API

```python
import requests

r = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "text": "안녕하세요. 테스트입니다.",
        "language": "Korean",
        "voice": "my_voice",
    },
    stream=True,
)
# r.iter_content() returns streaming PCM/WAV
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model to load |
| `TTS_MAX_BATCH` | `8` | Max requests per GPU batch |
| `TTS_BATCH_WAIT_MS` | `50` | Max wait (ms) before processing batch |
| `TTS_CUSTOM_VOICES` | `/app/custom_voices` | Custom voice profiles directory |
| `TTS_WARMUP_ON_START` | `false` | Warmup on startup |
| `TTS_INT8_QUANTIZE` | `false` | INT8 weight-only quantization |
| `MAX_CONCURRENT` | `64` | Max concurrent requests |

## Tools

| Script | Description |
|--------|-------------|
| `benchmark_rtf.py` | RTF benchmark with concurrent streaming requests |
| `concurrent_benchmark.py` | Concurrency scaling benchmark |
| `eval_utmos.py` | UTMOS speech quality evaluation (torch.hub) |
| `preprocess_refs.py` | Reference audio preprocessing (trim, normalize, pad) |

## Acknowledgments

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT) — Core vLLM-style inference engine
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) — Qwen3-TTS adaptation with continuous batching, paged attention, CUDA graphs, streaming
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) — The underlying TTS model

## License

MIT License. See [LICENSE](LICENSE) for details.
