"""Cache neural (Grave et al., 2017) na camada rasa: a linha de base da adaptação ao contexto.

Guarda pares (chave_i, próximo token real x_{i+1}) e mistura na previsão rasa a
distribuição dos tokens que vieram depois de chaves parecidas:

    p_cache(w) = Σ_{i: x_{i+1} = w} softmax_i(β · cos(key_t, key_i))
    p(w)       = (1 - λ) · p_k(w) + λ · p_cache(w)

Diferenças para o MYELIN:
- aprende com TODO token (o próximo token já foi lido), sem acordar o córtex;
- corrige no espaço dos tokens (uma distribuição), não no estado oculto.

Se ele empatar com o MYELIN, a memória rápida não precisa do professor profundo.
Atenção: na simulação, o "próximo token real" vem do texto; numa geração com
saída cedo, seria o token emitido, às vezes errado.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class NeuralCache:
    """mix(chaves, logits) -> log p misturado; add(chaves, próximos tokens) guarda os pares."""

    def __init__(
        self, beta: float = 20.0, lam: float = 0.1, capacity: int | None = 256, min_sim: float | None = None
    ) -> None:
        """min_sim: se dado, só mistura quando alguma chave guardada tem cos >= min_sim (como o KNNCache)."""
        self.beta, self.lam, self.capacity, self.min_sim = beta, lam, capacity, min_sim
        self.keys: torch.Tensor | None = None
        self.ids: torch.Tensor | None = None

    def mix(self, key: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Log-probabilidades (N, |V|) da mistura; sem pares guardados (ou nenhum familiar), log p_k."""
        log_p = F.log_softmax(logits, -1)
        if self.keys is None:
            return log_p
        sims = F.normalize(key, dim=-1) @ self.keys.T  # (N, M)
        weights = torch.softmax(self.beta * sims, dim=-1)
        p_cache = torch.zeros_like(log_p).scatter_add_(-1, self.ids.expand(len(key), -1), weights)
        keep = math.log1p(-self.lam) if self.lam < 1 else -math.inf
        mixed = torch.logaddexp(log_p + keep, (self.lam * p_cache).log())
        if self.min_sim is None:
            return mixed
        familiar = (sims.max(-1).values >= self.min_sim).unsqueeze(-1)
        return torch.where(familiar, mixed, log_p)

    def add(self, key: torch.Tensor, next_ids: torch.Tensor) -> None:
        """Guarda os pares com próximo token conhecido (ids >= 0), só os `capacity` últimos."""
        known = next_ids >= 0
        keys, ids = F.normalize(key[known], dim=-1), next_ids[known].to(key.device)
        self.keys = keys if self.keys is None else torch.cat([self.keys, keys])
        self.ids = ids if self.ids is None else torch.cat([self.ids, ids])
        if self.capacity is not None:
            self.keys, self.ids = self.keys[-self.capacity :], self.ids[-self.capacity :]

    def reset(self) -> None:
        self.keys, self.ids = None, None
