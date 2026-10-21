"""Tradutor P_k: lê o estado de uma camada intermediária na "língua" da camada final.

    P_k(h) = h + W_k h + b_k        (W_k = 0 e b_k = 0 no início = logit lens)

É o *tuned lens* (Belrose et al., 2023): uma transformação afim por camada,
treinada UMA vez, offline, para que lm_head(norm(P_k(h_k))) imite a
distribuição final do modelo. Depois fica congelada.

O Experimento 0 mostrou que, sem ele, as camadas rasas do Qwen concordam 0%
com a camada final e ainda ficam confiantes e erradas. Divisão de trabalho:
P_k aprende a tradução geral entre camadas; a Sinapse Rápida aprende os
atalhos do contexto atual, em cima dela.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TunedLens(nn.Module):
    """Um tradutor afim por camada 1..L-1 (a camada L já está na língua final)."""

    def __init__(self, n_layers: int, d_model: int) -> None:
        super().__init__()
        self.n_layers, self.d_model = n_layers, d_model
        self.translators = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_layers - 1))
        for translator in self.translators:
            nn.init.zeros_(translator.weight)
            nn.init.zeros_(translator.bias)

    def forward(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """Traduz h, saída da camada `layer` (1..L-1), para a língua da camada L."""
        return h + self.translators[layer - 1](h)

    def save(self, path: str | Path, **metadata) -> None:
        torch.save(
            {"n_layers": self.n_layers, "d_model": self.d_model, "state_dict": self.state_dict(), **metadata},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> TunedLens:
        checkpoint = torch.load(path, map_location=map_location)
        lens = cls(checkpoint["n_layers"], checkpoint["d_model"])
        lens.load_state_dict(checkpoint["state_dict"])  # copia para os pesos do módulo, criados na CPU
        return lens.to(map_location)
