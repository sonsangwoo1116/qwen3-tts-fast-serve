"""
BatchedStreamingEngine: single-engine concurrent streaming TTS.

Replaces multi-worker architecture. One model instance serves N concurrent
requests by batching Talker and Predictor step() calls.

Architecture:
  submit_streaming() → inbox queue → _engine_thread picks up
  _engine_thread runs tight loop:
    1) drain inbox (new requests)
    2) predictor.step() for all active predictor sequences (batched)
    3) talker.step_with_outputs() for all active talker sequences (batched)
    4) sleep if idle
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from qwen3_tts_engine.sampling_params import SamplingParams

logger = logging.getLogger("batched-engine")


@dataclass
class RequestState:
    """Per-request mutable state tracked by the engine."""
    request_id: str
    trailing_text_hiddens: torch.Tensor   # [1, T, D]
    tts_pad_embed: torch.Tensor           # [1, 1, D]
    generation_step: int = 0
    max_steps: int = 720
    chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    event_loop: asyncio.AbstractEventLoop = None
    finished: bool = False


class BatchedStreamingEngine:
    """Single-engine batched streaming TTS processor.

    Uses one Talker + one Predictor LLM instance. All concurrent requests
    share the same GPU via batched step() calls.
    """

    # Max generation steps per request (safety limit for TTS).
    # At 12Hz, 720 steps = 60 seconds of audio. Beyond this, force-finish.
    MAX_GENERATION_STEPS = 720

    def __init__(self, interface, max_concurrent: int = 64):
        self._iface = interface
        self._max_concurrent = max_concurrent
        self._device = interface.device

        # Model references (no copies)
        self._talker = interface.talker_llm
        self._predictor = interface.predictor_llm
        self._input_embedding = interface.input_embedding
        self._predictor_weights = interface.predictor_input_embeddings  # nn.ModuleList[16]

        # Engine state
        self._active: Dict[str, RequestState] = {}
        self._inbox: list = []  # protected by _inbox_lock
        self._inbox_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background engine thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._engine_loop, daemon=True, name="batched-engine",
        )
        self._thread.start()
        logger.info("BatchedStreamingEngine started")

    def stop(self):
        """Stop the background engine thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BatchedStreamingEngine stopped")

    @torch.inference_mode()
    def submit_streaming(
        self,
        text: str,
        language: str,
        voice_prompt: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Queue:
        """Submit a new streaming TTS request. Returns an asyncio.Queue
        that will receive PCM-ready codec chunk lists, with None as sentinel."""
        request_id = str(uuid.uuid4())

        # Prepare inputs on the calling thread (short CPU+GPU work ~10ms)
        prepared = self._iface.prepare_voice_clone_inputs(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompt,
            non_streaming_mode=True,
        )

        # Estimate max generation steps from text length
        # Korean: ~0.2-0.3s per char at 12Hz → ~3-4 frames per char
        # Use 12 frames/char with min 60 (5s) and max 720 (60s)
        max_steps = max(60, min(3600, len(text) * 8))

        chunk_queue: asyncio.Queue = asyncio.Queue()
        state = RequestState(
            request_id=request_id,
            trailing_text_hiddens=prepared["trailing_text_hiddens"],
            tts_pad_embed=prepared["tts_pad_embed"],
            max_steps=max_steps,
            chunk_queue=chunk_queue,
            event_loop=loop,
        )

        inbox_item = {
            "request_id": request_id,
            "inputs_embeds": prepared["inputs_embeds"],
            "state": state,
        }

        with self._inbox_lock:
            self._inbox.append(inbox_item)

        return chunk_queue

    @torch.inference_mode()
    def submit_streaming_custom(
        self,
        text: str,
        language: str,
        speaker: str,
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Queue:
        """Submit a streaming TTS request using built-in speaker (no voice clone)."""
        request_id = str(uuid.uuid4())

        prepared = self._iface.prepare_custom_voice_inputs(
            text=text,
            language=language,
            speaker=speaker,
            non_streaming_mode=True,
        )

        max_steps = max(60, min(3600, len(text) * 8))

        chunk_queue: asyncio.Queue = asyncio.Queue()
        state = RequestState(
            request_id=request_id,
            trailing_text_hiddens=prepared["trailing_text_hiddens"],
            tts_pad_embed=prepared["tts_pad_embed"],
            max_steps=max_steps,
            chunk_queue=chunk_queue,
            event_loop=loop,
        )

        inbox_item = {
            "request_id": request_id,
            "inputs_embeds": prepared["inputs_embeds"],
            "state": state,
        }

        with self._inbox_lock:
            self._inbox.append(inbox_item)

        return chunk_queue

    def submit_custom(
        self,
        text: str,
        language: str,
        speaker: str,
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Future:
        """Submit a non-streaming TTS request using built-in speaker."""
        fut = loop.create_future()
        chunk_q = self.submit_streaming_custom(text, language, speaker, loop)

        async def _collect():
            try:
                all_codes = []
                while True:
                    chunk = await chunk_q.get()
                    if chunk is None:
                        break
                    all_codes.append(chunk)
                if all_codes:
                    codes_tensor = torch.tensor(all_codes, dtype=torch.long)
                    wavs, sr = self._iface.speech_tokenizer.decode(
                        [{"audio_codes": codes_tensor}],
                    )
                    import io, soundfile as sf
                    buf = io.BytesIO()
                    sf.write(buf, wavs[0], sr, format="WAV")
                    fut.set_result(buf.getvalue())
                else:
                    fut.set_result(b"")
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_collect()))
        return fut

    def submit(
        self,
        text: str,
        language: str,
        voice_prompt: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Future:
        """Submit a non-streaming TTS request. Returns a Future that resolves
        to wav bytes when generation completes."""
        fut = loop.create_future()
        chunk_q = self.submit_streaming(text, language, voice_prompt, loop)

        # Extract ref_code for decoder context (ICL mode only)
        ref_code = None
        if voice_prompt.get("icl_mode") and voice_prompt.get("ref_code") is not None:
            ref_code = voice_prompt["ref_code"]

        # Wrap: collect all chunks, decode to wav, resolve future
        async def _collect():
            try:
                all_codes = []
                while True:
                    chunk = await chunk_q.get()
                    if chunk is None:
                        break
                    all_codes.append(chunk)
                if all_codes:
                    codes_tensor = torch.tensor(all_codes, dtype=torch.long)

                    # Prepend ref_code for decoder context
                    if ref_code is not None:
                        ref_code_t = ref_code.to(codes_tensor.device)
                        # ref_code is [time, 16], same shape as codes_tensor
                        ref_len = ref_code_t.shape[0]
                        codes_for_decode = torch.cat([ref_code_t, codes_tensor], dim=0)
                    else:
                        ref_len = 0
                        codes_for_decode = codes_tensor

                    wavs, sr = self._iface.speech_tokenizer.decode(
                        [{"audio_codes": codes_for_decode}],
                    )

                    # Trim reference portion
                    audio = wavs[0]
                    if ref_len > 0:
                        total_frames = codes_for_decode.shape[0]
                        cut_samples = int(ref_len / total_frames * len(audio))
                        audio = audio[cut_samples:]

                    import io, soundfile as sf
                    buf = io.BytesIO()
                    sf.write(buf, audio, sr, format="WAV")
                    fut.set_result(buf.getvalue())
                else:
                    fut.set_result(b"")
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_collect()))
        return fut

    # ── Private: engine loop ─────────────────────────────────

    def _drain_inbox(self):
        """Move new requests from inbox into active set and enqueue talker prefill."""
        with self._inbox_lock:
            items = self._inbox[:]
            self._inbox.clear()

        talker_sp = SamplingParams(temperature=0.8, max_tokens=1, repetition_penalty=1.05)
        for item in items:
            rid = item["request_id"]
            state = item["state"]
            embeds = item["inputs_embeds"]
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            self._active[rid] = state
            self._talker.add_request([embeds], talker_sp, request_id=rid)
            logger.info(f"[engine] new request {rid[:8]}...")

    def _finish_request(self, request_id: str):
        """Mark request as done and send sentinel to chunk queue."""
        state = self._active.pop(request_id, None)
        if state is not None:
            state.finished = True
            state.event_loop.call_soon_threadsafe(
                state.chunk_queue.put_nowait, None,
            )
            logger.info(f"[engine] finished {request_id[:8]} steps={state.generation_step}")

    def _engine_loop(self):
        """Background thread. Runs Talker/Predictor step() in a tight loop."""
        talker_sp = SamplingParams(temperature=0.8, max_tokens=1, repetition_penalty=1.05)
        predictor_sp = SamplingParams(temperature=0.2, max_tokens=17)

        # Pending predictor results: rid → {last_id, last_id_hidden}
        pending_predictor: Dict[str, dict] = {}

        while self._running:
            try:
                # 1) Drain inbox
                self._drain_inbox()

                has_pred = bool(
                    self._predictor.scheduler.waiting
                    or self._predictor.scheduler.running
                )
                has_talk = bool(
                    self._talker.scheduler.waiting
                    or self._talker.scheduler.running
                )

                if not has_pred and not has_talk:
                    time.sleep(0.001)
                    continue

                # 2) Predictor step (if any sequences waiting/running)
                if has_pred:
                    outputs, _ = self._predictor.step()
                    if outputs:
                        batch_items = []
                        for request_id, seq_id, pred_token_ids in outputs:
                            state = self._active.get(request_id)
                            pending = pending_predictor.pop(request_id, None)
                            if not state or not pending:
                                continue
                            # Deliver codec chunk to caller (exactly 16 codebooks: 1 + 15)
                            codebook_ids = [pending["last_id"]] + pred_token_ids[:15]
                            state.event_loop.call_soon_threadsafe(
                                state.chunk_queue.put_nowait, codebook_ids,
                            )
                            batch_items.append({
                                "request_id": request_id,
                                "last_id": pending["last_id"],
                                "pred_token_ids": pred_token_ids[:15],
                                "state": state,
                            })
                        # Vectorized next-talker-embeds computation
                        if batch_items:
                            embeds_map = self._compute_next_talker_embeds_batch(batch_items)
                            for rid, next_embeds in embeds_map.items():
                                self._active[rid].generation_step += 1
                                self._talker.add_request(
                                    [next_embeds], talker_sp, request_id=rid,
                                )

                # 3) Talker step (always runs in same iteration — no starvation)
                #    Re-check since predictor may have added new talker requests above
                has_talk = bool(
                    self._talker.scheduler.waiting
                    or self._talker.scheduler.running
                )
                if has_talk:
                    _, _, outputs_all = self._talker.step_with_outputs()
                    # First pass: filter finished, collect continuations
                    cont_items = []
                    cont_last_ids = []
                    for request_id, seq_id, token_ids, hidden_states, is_finished in outputs_all:
                        state = self._active.get(request_id)
                        if not state:
                            continue
                        last_id = token_ids[-1]
                        if (last_id == 2150 or is_finished
                                or state.generation_step >= state.max_steps):
                            reason = "eos" if last_id == 2150 else ("max_steps" if state.generation_step >= state.max_steps else "scheduler_finish")
                            logger.info(f"[engine] stopping {request_id[:8]} reason={reason} last_id={last_id} step={state.generation_step}")
                            self._talker.clear_request(request_id)
                            self._finish_request(request_id)
                            continue
                        cont_items.append((request_id, last_id, hidden_states))
                        cont_last_ids.append(last_id)
                    # Batched embedding lookup (single kernel instead of N)
                    if cont_items:
                        _t0 = time.perf_counter()
                        all_id_hiddens = self._input_embedding(
                            torch.tensor(cont_last_ids, device=self._device)
                        )  # (N, hidden_size)
                        for idx, (request_id, last_id, hidden_states) in enumerate(cont_items):
                            last_id_hidden = all_id_hiddens[idx].unsqueeze(0).unsqueeze(0)
                            last_hidden_state = hidden_states.unsqueeze(0).unsqueeze(0)
                            pred_embeds = torch.cat(
                                (last_hidden_state, last_id_hidden), dim=1,
                            )
                            pending_predictor[request_id] = {
                                "last_id": last_id,
                                "last_id_hidden": last_id_hidden,
                            }
                            self._predictor.add_request(
                                [pred_embeds], predictor_sp, request_id=request_id,
                            )

            except Exception:
                import traceback
                logger.error(f"[engine] error:\n{traceback.format_exc()}")
                # Abort all active requests to avoid cascading errors
                for rid in list(self._active.keys()):
                    self._talker.clear_request(rid)
                    self._predictor.clear_request(rid)
                    self._finish_request(rid)
                pending_predictor.clear()
                time.sleep(0.01)

    @torch.inference_mode()
    def _compute_next_talker_embeds_batch(
        self, items: List[dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute next talker input embeddings for N finished predictor seqs.

        Vectorized: one batched embedding lookup + 15 batched index selects
        instead of 15*N individual embedding calls.
        """
        N = len(items)
        device = self._device

        # Codebook-0: batched lookup
        cb0_ids = torch.tensor(
            [it["last_id"] for it in items], device=device,
        )
        cb0_embeds = self._input_embedding(cb0_ids)  # [N, D]

        # Codebooks 1-15: batched index select
        pred_ids = torch.tensor(
            [it["pred_token_ids"] for it in items], device=device,
        )  # [N, 15]

        cb_sum = torch.zeros_like(cb0_embeds)  # [N, D]
        for i in range(15):
            cb_sum = cb_sum + self._predictor_weights[i](pred_ids[:, i])  # [N, D]

        total = cb0_embeds + cb_sum  # [N, D]

        results = {}
        for idx, item in enumerate(items):
            state = item["state"]
            next_embed = total[idx].unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            gs = state.generation_step
            if gs < state.trailing_text_hiddens.shape[1]:
                next_embed = next_embed + state.trailing_text_hiddens[:, gs].unsqueeze(1)
            else:
                next_embed = next_embed + state.tts_pad_embed
            results[item["request_id"]] = next_embed
        return results
