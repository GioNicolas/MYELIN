"""Controles da memória rápida (pedidos pela revisão externa do Experimento 6).

Se um destes controles igualar o KNNCache, o ganho dele não vem de reconhecer
a situação pela chave:
  - LastCorrection: aplica a última correção ensinada, para qualquer chave
    (continuidade local);
  - MeanCorrection: aplica a média de todas as correções ensinadas (correção
    global, como o viés lento);
  - ShuffledKNN: igual ao KNNCache (mesma porta de familiaridade, mesmas
    correções guardadas), mas com o par chave → correção embaralhado a cada
    consulta.

Mesma interface da FastSynapse e do KNNCache: forward, consolidate, decay.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from myelin.knn_cache import KNNCache


class LastCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last: torch.Tensor | None = None

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(key) if self.last is None else self.last.expand_as(key)

    @torch.no_grad()
    def consolidate(self, h_in: torch.Tensor, residual: torch.Tensor, mask=None, decay: bool = True) -> None:
        self.last = (residual + self(h_in))[-1].clone()  # correção absoluta = resíduo + o que já aplicava

    def decay(self) -> None:
        pass


class MeanCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.total: torch.Tensor | None = None
        self.count = 0

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(key) if self.total is None else (self.total / self.count).expand_as(key)

    @torch.no_grad()
    def consolidate(self, h_in: torch.Tensor, residual: torch.Tensor, mask=None, decay: bool = True) -> None:
        corrections = residual + self(h_in)
        self.total = corrections.sum(0) if self.total is None else self.total + corrections.sum(0)
        self.count += len(corrections)

    def decay(self) -> None:
        pass


class ShuffledKNN(KNNCache):
    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = torch.Generator().manual_seed(seed)
        self._applied: torch.Tensor | None = None

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            self._applied = torch.zeros_like(key)
            return self._applied
        sims = F.normalize(key, dim=-1) @ torch.stack(self.keys).T
        weights = torch.softmax(self.beta * sims, dim=-1)
        familiar = (sims.max(-1).values >= self.min_sim).unsqueeze(-1)
        perm = torch.randperm(len(self.corrections), generator=self.generator).tolist()
        self._applied = familiar * (weights @ torch.stack([self.corrections[i] for i in perm]))
        return self._applied

    @torch.no_grad()
    def consolidate(self, h_in: torch.Tensor, residual: torch.Tensor, mask=None, decay: bool = True) -> None:
        """Guarda a correção verdadeira, resíduo + o que foi aplicado na ÚLTIMA consulta (chame forward antes)."""
        keys = h_in.reshape(-1, h_in.shape[-1])
        applied = getattr(self, "_applied", None)
        if applied is None or applied.shape != keys.shape:
            applied = torch.zeros_like(keys)
        self.keys += list(F.normalize(keys, dim=-1))
        self.corrections += list(residual.reshape(-1, residual.shape[-1]) + applied)
        if self.capacity is not None and len(self.keys) > self.capacity:
            self.keys, self.corrections = self.keys[-self.capacity :], self.corrections[-self.capacity :]
        self._applied = None
