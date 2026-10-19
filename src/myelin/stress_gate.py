"""Peça 2 do MYELIN: o Medidor de Estresse S_t e o Portão de Decisão.

    S_t = -(1 / log|V|) · Σ_v p_t^(k)(v) log p_t^(k)(v)     ∈ [0, 1]

S_t <= τ  -> reflexo consolidado: aceita a previsão da camada rasa k.
S_t >  τ  -> dúvida: acorda o Córtex Profundo (camadas k+1..L).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def entropy_stress(logits: torch.Tensor) -> torch.Tensor:
    """Entropia normalizada da distribuição sobre o vocabulário, forma (...,)."""
    log_p = F.log_softmax(logits, dim=-1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    return entropy / math.log(logits.shape[-1])


def margin_stress(logits: torch.Tensor) -> torch.Tensor:
    """1 - (p_top1 - p_top2): alto quando os dois melhores candidatos empatam."""
    top2 = F.softmax(logits, dim=-1).topk(2, dim=-1).values
    return 1.0 - (top2[..., 0] - top2[..., 1])


_METRICS = {"entropy": entropy_stress, "margin": margin_stress}


class StressGate:
    """Decide, token a token, se a camada rasa basta ou se o córtex deve acordar."""

    def __init__(self, tau: float = 0.3, metric: str = "entropy") -> None:
        if metric not in _METRICS:
            raise ValueError(f"métrica desconhecida {metric!r}; use {sorted(_METRICS)}")
        self.tau = tau
        self.metric = metric
        self._stress_fn = _METRICS[metric]

    def __call__(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (S_t, acordar_cortex) para logits de forma (..., |V|)."""
        stress = self._stress_fn(logits)
        return stress, stress > self.tau
