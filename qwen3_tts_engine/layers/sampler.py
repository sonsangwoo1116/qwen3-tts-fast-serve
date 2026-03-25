import torch
from torch import nn
import time

class Sampler(nn.Module):

    def __init__(self, top_k: int = 50):
        super().__init__()
        self._top_k = top_k

    def apply_temperature(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.float().div_(temperatures.unsqueeze(dim=1))

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        prev_token_ids: list[list[int]],
        penalties: torch.Tensor,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits.

        For each sequence in the batch, tokens that appeared in prev_token_ids
        get their logits divided by penalty (if positive) or multiplied (if negative).
        Returns a new tensor (clone) to avoid InferenceMode issues.
        """
        logits = logits.clone()
        for i, (token_ids, penalty) in enumerate(zip(prev_token_ids, penalties)):
            if penalty == 1.0 or not token_ids:
                continue
            unique_ids = list(set(token_ids))
            ids_tensor = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)
            score = logits[i, ids_tensor]
            score = torch.where(score > 0, score / penalty, score * penalty)
            logits[i, ids_tensor] = score
        return logits

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        indices_to_remove = logits < torch.topk(logits, self._top_k)[0][..., -1, None]
        top_k_logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        probs = torch.softmax(top_k_logits, dim=-1)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return sample_tokens
