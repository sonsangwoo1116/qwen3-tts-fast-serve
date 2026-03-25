import logging
from collections import deque

from qwen3_tts_engine.config import Config
from qwen3_tts_engine.engine.sequence import Sequence, SequenceStatus
from qwen3_tts_engine.engine.block_manager import BlockManager

logger = logging.getLogger("scheduler")


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                print(f"num_batched_tokens: {num_batched_tokens}")
                print(f"len(seq): {len(seq)}")
                print(f"max_num_batched_tokens: {self.max_num_batched_tokens}")
                print(f"block_manager.can_allocate(seq): {self.block_manager.can_allocate(seq)}")
                print(f"seq.num_blocks: {seq.num_blocks}")
                print(f"seq.num_tokens: {seq.num_tokens}")
                print(f"Breaking prefill because of batch size or block manager")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
    
        if not scheduled_seqs:
            # All sequences were preempted — they are back in waiting.
            # Return empty to let the caller retry next iteration.
            return [], False
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        logger.warning(
            f"PREEMPTION: seq_id={seq.seq_id} request_id={getattr(seq, 'request_id', None)} "
            f"tokens={seq.num_tokens} blocks={seq.num_blocks} "
            f"free_blocks={len(self.block_manager.free_block_ids)}"
        )
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id, last_hidden_state=None)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
