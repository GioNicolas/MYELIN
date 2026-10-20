"""Memórias rápidas auxiliares: um viés online e a soma de várias memórias.

OnlineBias: a mesma correção para todo token, aprendida ao vivo (ignora a
chave). Pega o que é global, como o erro de idioma do tradutor P_k.

StackedMemory: Δq_t = m_1(key_t) + m_2(key_t) + ... Na consolidação, m_1
recebe o resíduo inteiro; m_2 recebe o que sobrou depois de m_1 se ajustar
naquele token, e assim por diante. Ex.: OnlineBias seguido de um kNN (o que é
específico de cada situação).
"""

from __future__ import annotations

import torch
from torch import nn


class OnlineBias(nn.Module):
    """b <- γ b + η · média(resíduo); Δq_t = b para todo token."""

    def __init__(self, d_out: int, eta: float = 0.5, gamma: float = 1.0) -> None:
        super().__init__()
        self.eta, self.gamma = eta, gamma
        self.register_buffer("b", torch.zeros(d_out))

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return self.b.expand(*key.shape[:-1], -1)

    @torch.no_grad()
    def consolidate(
        self, h_in: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor | None = None, decay: bool = True
    ) -> None:
        r = residual.reshape(-1, residual.shape[-1])
        if mask is not None:
            r = r[mask.reshape(-1)]
        if r.shape[0] == 0:
            return
        if decay:
            self.b.mul_(self.gamma)
        self.b.add_(r.mean(0), alpha=self.eta)

    @torch.no_grad()
    def decay(self) -> None:
        self.b.mul_(self.gamma)


class StackedMemory(nn.Module):
    """Mesma interface da FastSynapse (forward, consolidate, decay)."""

    def __init__(self, *memories: nn.Module) -> None:
        super().__init__()
        self.memories = nn.ModuleList(memories)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return sum(memory(key) for memory in self.memories)

    @torch.no_grad()
    def consolidate(
        self, h_in: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor | None = None, decay: bool = True
    ) -> None:
        for memory in self.memories:
            before = memory(h_in).clone()  # clone: a saída pode ser uma view do estado interno
            memory.consolidate(h_in, residual, mask, decay=decay)
            residual = residual - (memory(h_in) - before)

    def decay(self) -> None:
        for memory in self.memories:
            memory.decay()
