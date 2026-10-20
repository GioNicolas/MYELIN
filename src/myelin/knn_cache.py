"""Memória exata de pares (chave, correção): o teto não paramétrico da Sinapse Rápida.

Em vez de comprimir tudo numa matriz B de posto r, guarda cada correção que o
córtex ensinou e, para uma chave nova, soma as correções das chaves parecidas:

    Δq_t = Σ_i softmax_i(β · cos(key_t, key_i)) · c_i      (só se max_i cos >= min_sim)
    c_i  = v_i - P_k(h_i^(k))                              (correção absoluta do token i)

Ela cresce com o texto, então NÃO é o MYELIN: serve para separar duas
perguntas. Se nem ela melhora a camada rasa, a correção do córtex não é
previsível a partir da chave; se ela melhora e a sinapse B não, o problema está
na compressão de posto baixo (colisão de chaves).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KNNCache(nn.Module):
    """Mesma interface da FastSynapse (forward, consolidate, decay, reset)."""

    def __init__(self, beta: float = 20.0, min_sim: float = 0.5) -> None:
        super().__init__()
        self.beta, self.min_sim = beta, min_sim
        self.keys: list[torch.Tensor] = []
        self.corrections: list[torch.Tensor] = []

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """Correção Δq para chaves de forma (N, d); zero se nada parecido foi guardado."""
        if not self.keys:
            return torch.zeros_like(key)
        sims = F.normalize(key, dim=-1) @ torch.stack(self.keys).T  # (N, M)
        weights = torch.softmax(self.beta * sims, dim=-1)
        familiar = (sims.max(-1).values >= self.min_sim).unsqueeze(-1)
        return familiar * (weights @ torch.stack(self.corrections))

    @torch.no_grad()
    def consolidate(
        self, h_in: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor | None = None, decay: bool = True
    ) -> None:
        """Guarda c_i = resíduo + correção atual (= alvo - estado raso sem memória)."""
        keep = slice(None) if mask is None else mask.reshape(-1)
        keys, residual = h_in.reshape(-1, h_in.shape[-1])[keep], residual.reshape(-1, residual.shape[-1])[keep]
        corrections = residual + self(keys)
        self.keys += list(F.normalize(keys, dim=-1))
        self.corrections += list(corrections)

    def decay(self) -> None:
        """Sem homeostase: a memória exata não esquece."""

    def reset(self) -> None:
        self.keys, self.corrections = [], []

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.corrections)
