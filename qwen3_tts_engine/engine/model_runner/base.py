import os
import pickle
import torch
import torch.distributed as dist
from safetensors.torch import load_file
import json
from typing import Optional
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from qwen3_tts_engine.config import Config
from qwen3_tts_engine.engine.sequence import Sequence
from qwen3_tts_engine.layers.sampler import Sampler
from qwen3_tts_engine.utils.context import set_context, get_context, reset_context
from qwen3_tts_engine.config import Qwen3TTSConfig
from qwen3_tts_engine.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from qwen3_tts_engine.models.qwen3_tts_predictor import Qwen3TTSCodePredictorForCausalLM

MODEL_TYPE_MAPPING = {
    "talker": Qwen3TTSTalkerForCausalLM,
    "predictor": Qwen3TTSCodePredictorForCausalLM,
}


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if not dist.is_initialized():
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        # torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
                
    def post_init(self, rank: int):
        default_dtype = torch.get_default_dtype()
        self.sampler = Sampler()
        self._init_decode_buffers()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def load_model(self, config: Config):
        ...

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([], input_embeds=torch.zeros(1, 8, self.model_config.hidden_size)) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = self.model_config
        torch_dtype = torch.bfloat16
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch_dtype.itemsize
        # Use fraction of FREE memory (not total) so multiple workers can share a GPU
        available = free * config.gpu_memory_utilization
        config.num_kvcache_blocks = max(1, int(available) // block_bytes)
        assert config.num_kvcache_blocks > 0, (
            f"KV cache: {config.num_kvcache_blocks} blocks, "
            f"free={free/1e9:.1f}GB, util={config.gpu_memory_utilization}"
        )
        import logging
        logging.getLogger("kv-cache").info(
            f"KV cache allocated: {config.num_kvcache_blocks} blocks, "
            f"block_size={self.block_size}, layers={hf_config.num_hidden_layers}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"block_bytes={block_bytes}, total={config.num_kvcache_blocks * block_bytes / 1e9:.2f}GB, "
            f"free_gpu={free/1e9:.2f}GB, util={config.gpu_memory_utilization}"
        )
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _init_decode_buffers(self):
        """Pre-allocate pinned CPU and CUDA buffers for prepare_decode/prepare_sample."""
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

        self._dbuf_cpu_input_ids = torch.zeros(max_bs, dtype=torch.int64, device='cpu').pin_memory()
        self._dbuf_cpu_positions = torch.zeros(max_bs, dtype=torch.int64, device='cpu').pin_memory()
        self._dbuf_cpu_slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device='cpu').pin_memory()
        self._dbuf_cpu_context_lens = torch.zeros(max_bs, dtype=torch.int32, device='cpu').pin_memory()
        self._dbuf_cpu_block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32, device='cpu').pin_memory()
        self._dbuf_cpu_temperatures = torch.zeros(max_bs, dtype=torch.float32, device='cpu').pin_memory()

        self._dbuf_gpu_input_ids = torch.zeros(max_bs, dtype=torch.int64, device='cuda')
        self._dbuf_gpu_positions = torch.zeros(max_bs, dtype=torch.int64, device='cuda')
        self._dbuf_gpu_slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device='cuda')
        self._dbuf_gpu_context_lens = torch.zeros(max_bs, dtype=torch.int32, device='cuda')
        self._dbuf_gpu_block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32, device='cuda')
        self._dbuf_gpu_temperatures = torch.zeros(max_bs, dtype=torch.float32, device='cuda')

        # Block table cache for consecutive decode steps
        self._bt_cache_valid = False
        self._bt_cache_bs = 0

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        input_embeds = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            input_embeds.extend(seq.input_embeds[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        input_embeds = torch.cat([e if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds], dim=0).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)
        else:
            input_embeds = input_embeds.cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, input_embeds, positions

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        for i, seq in enumerate(seqs):
            self._dbuf_cpu_input_ids[i] = seq.last_token
            self._dbuf_cpu_positions[i] = len(seq) - 1
            self._dbuf_cpu_context_lens[i] = len(seq)
            self._dbuf_cpu_slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1

        self._dbuf_gpu_input_ids[:bs].copy_(self._dbuf_cpu_input_ids[:bs], non_blocking=True)
        self._dbuf_gpu_positions[:bs].copy_(self._dbuf_cpu_positions[:bs], non_blocking=True)
        self._dbuf_gpu_slot_mapping[:bs].copy_(self._dbuf_cpu_slot_mapping[:bs], non_blocking=True)
        self._dbuf_gpu_context_lens[:bs].copy_(self._dbuf_cpu_context_lens[:bs], non_blocking=True)

        # Block table cache: reuse if same batch size and cache is valid
        if self._bt_cache_valid and self._bt_cache_bs == bs:
            block_tables = self._dbuf_gpu_block_tables[:bs, :self._bt_cache_max_len]
        else:
            max_len = max(len(seq.block_table) for seq in seqs)
            for i, seq in enumerate(seqs):
                bt = seq.block_table
                for j, b in enumerate(bt):
                    self._dbuf_cpu_block_tables[i, j] = b
                if len(bt) < max_len:
                    self._dbuf_cpu_block_tables[i, len(bt):max_len] = -1
            self._dbuf_gpu_block_tables[:bs, :max_len].copy_(
                self._dbuf_cpu_block_tables[:bs, :max_len], non_blocking=True)
            block_tables = self._dbuf_gpu_block_tables[:bs, :max_len]
            self._bt_cache_valid = True
            self._bt_cache_bs = bs
            self._bt_cache_max_len = max_len

        set_context(False, slot_mapping=self._dbuf_gpu_slot_mapping[:bs],
                    context_lens=self._dbuf_gpu_context_lens[:bs],
                    block_tables=block_tables)
        return self._dbuf_gpu_input_ids[:bs], self._dbuf_gpu_positions[:bs]

    def prepare_sample(self, seqs: list[Sequence]):
        bs = len(seqs)
        for i, seq in enumerate(seqs):
            self._dbuf_cpu_temperatures[i] = seq.temperature
        self._dbuf_gpu_temperatures[:bs].copy_(self._dbuf_cpu_temperatures[:bs], non_blocking=True)
        return self._dbuf_gpu_temperatures[:bs]

    def prepare_repetition_penalty(self, seqs: list[Sequence]):
        """Extract per-sequence repetition penalty info for the sampler."""
        penalties = []
        prev_tokens = []
        need_penalty = False
        for seq in seqs:
            p = getattr(seq, 'repetition_penalty', 1.0)
            penalties.append(p)
            if p != 1.0:
                need_penalty = True
                prev_tokens.append(seq.token_ids[:])
            else:
                prev_tokens.append([])
        if not need_penalty:
            return None, None
        penalties_t = torch.tensor(penalties, dtype=torch.float32, device="cuda")
        return prev_tokens, penalties_t

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, input_embeds: Optional[torch.Tensor] = None):
        model_input = input_embeds if input_embeds is not None else input_ids
        
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or input_embeds is not None:
            return self.model.compute_logits(self.model(model_input, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            self._bt_cache_valid = False
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, input_embeds)
        if self.rank == 0:
            prev_tokens, penalties = self.prepare_repetition_penalty(seqs)
            if prev_tokens is not None:
                logits = self.sampler.apply_repetition_penalty(logits, prev_tokens, penalties)
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
