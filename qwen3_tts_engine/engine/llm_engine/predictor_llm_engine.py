import torch
from time import perf_counter
from tqdm.auto import tqdm

from qwen3_tts_engine.config import Config
from qwen3_tts_engine.engine.llm_engine.base import LLMEngine
from qwen3_tts_engine.engine.scheduler import Scheduler
from qwen3_tts_engine.engine.sequence import Sequence
from qwen3_tts_engine.engine.model_runner.predictor_model_runner import PredictorSequence
from qwen3_tts_engine.sampling_params import SamplingParams
from qwen3_tts_engine.engine.model_runner.predictor_model_runner import PredictorModelRunner


class PredictorScheduler(Scheduler):
    def __init__(self, config: Config):
        super().__init__(config)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def clear_request(self, request_id: str):
        if request_id in self.request_id_to_seq:
            seq = self.request_id_to_seq.pop(request_id)
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)
            if seq in self.waiting:
                self.waiting.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        super().postprocess(seqs, token_ids)
        for seq in seqs:
            if seq.is_finished and seq.request_id is not None:
                self.request_id_to_seq.pop(seq.request_id, None)


class PredictorLLMEngine(LLMEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model_runner = PredictorModelRunner(self.config, 0, self.events)
        self.scheduler = PredictorScheduler(self.config)

    def add_request(
        self,
        inputs_embeds: list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams],
        request_id: str | None = None,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs_embeds)
        # Clean up any orphaned previous predictor sequence for this request
        if request_id is not None and request_id in self.scheduler.request_id_to_seq:
            self.scheduler.clear_request(request_id)
        for inp_embeds, sp in zip(inputs_embeds, sampling_params):
            seq = PredictorSequence([], input_embeds=inp_embeds, sampling_params=sp, generation_steps=0, request_id=request_id)
            if request_id is not None:
                self.scheduler.request_id_to_seq[request_id] = seq
            self.scheduler.add(seq)

    def clear_request(self, request_id: str):
        self.scheduler.clear_request(request_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        for seq in seqs:
            seq.generation_steps += 1

        outputs = [(seq.request_id, seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def step_all(self):
        """Run step() repeatedly until all waiting/running sequences finish.

        Returns accumulated outputs from all steps.
        Avoids engine-loop round-trip overhead between predictor decode steps.
        """
        all_outputs = []
        while self.scheduler.waiting or self.scheduler.running:
            outputs, _ = self.step()
            if outputs:
                all_outputs.extend(outputs)
        return all_outputs

    def generate(
        self,
        inputs_embeds: list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        request_id: str | None = None,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(inputs_embeds), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs_embeds)
        for inp_embeds, sp in zip(inputs_embeds, sampling_params):
            self.add_request(inp_embeds, sp, request_id=request_id)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for req_id, seq_id, token_ids in output:
                outputs[seq_id] = (req_id, token_ids, None)
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        result = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "hidden_states": hidden_states}
            for _req_id, token_ids, hidden_states in outputs
        ]
        if request_id is not None:
            result = [r for r, (rid, _, _) in zip(result, outputs) if rid == request_id]
        if use_tqdm:
            pbar.close()
        return result
