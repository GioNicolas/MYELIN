"""A camada rasa como rascunho de decodificação especulativa (auto-especulação).

A camada k (lida por P_k, com a memória rápida) propõe o próximo token; o
modelo completo verifica. A saída final é IDÊNTICA à do modelo completo, e a
velocidade depende só da taxa de aceitação do rascunho:

    gulosa:      a_t = [argmax p_k == argmax p_L]
    amostragem:  α_t = Σ_v min(p_k(v), p_L(v)) = 1 - TV(p_k, p_L)   (Leviathan et al., 2023)

Como o modelo completo verifica todo token, o professor está sempre
disponível: a memória aprende com todos, depois de cada rascunho ser medido.

Speedup esperado (aceitações independentes, rascunho de γ tokens por ciclo):

    E(γ)  = (1 - a^(γ+1)) / (1 - a)                  tokens por ciclo
    custo = γ · c_rascunho + 1                       em unidades de 1 passo do modelo completo
    c_rascunho = (k + H) / (L + H)

H = custo da cabeça d × |V| em camadas: o rascunho paga a cabeça em todo token.
A verificação custa um passo inteiro: as camadas 1..k dos γ primeiros tokens
são reaproveitadas do rascunho, mas o último token rascunhado ainda precisa
passar pelas camadas 1..k (k) antes de as γ+1 posições passarem juntas pelas
camadas k+1..L e pela cabeça (L - k + H). Até a revisão externa do Experimento 6,
esta conta omitia esse k (ver docs/resultados-exp6.md, "Correção").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from myelin.neural_cache import NeuralCache
from myelin.wrapper import LayerStates, Memory, Readout


@dataclass
class DraftLog:
    """Registro token a token de um rascunho; todos os campos têm forma (T,)."""

    greedy: torch.Tensor  # top-1 do rascunho == top-1 do modelo completo
    alpha: torch.Tensor  # Σ min(p_k, p_L): aceitação esperada na amostragem
    shallow_nll: torch.Tensor  # -log p_k(próximo token real); NaN no último


@torch.no_grad()
def draft(
    states: LayerStates,
    readout: Readout,
    memory: Memory | None = None,
    cache: NeuralCache | None = None,
    learn: bool = True,
    chunk: int = 128,
) -> DraftLog:
    """Percorre o texto token a token: mede o rascunho e, se `learn`, a memória aprende com o verificador.

    A distribuição final vem de `states.target` (h_L / rms(h_L)): o RMSNorm não
    depende da escala, então readout(target) dá os logits do modelo completo.
    """
    T = states.key.shape[0]
    greedy = torch.zeros(T, dtype=torch.bool)
    alpha, shallow_nll = torch.zeros(T), torch.full((T,), float("nan"))

    def rms(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()

    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        p_final = F.softmax(readout(states.target[a:b]).float(), -1)
        for t in range(a, b):
            key, shallow = states.key[t : t + 1], states.shallow[t : t + 1]
            q = shallow if memory is None else shallow + memory(key)
            logits = readout(q).float()
            log_p = cache.mix(key, logits) if cache is not None else F.log_softmax(logits, -1)

            greedy[t] = (log_p.argmax(-1) == states.final_top[t]).item()
            alpha[t] = torch.minimum(log_p.exp(), p_final[t - a : t - a + 1]).sum().item()
            target_id = states.next_ids[t].item()
            if target_id >= 0:
                shallow_nll[t] = -log_p[0, target_id].item()

            if learn and memory is not None:
                residual = rms(shallow) * states.target[t : t + 1] - q
                memory.consolidate(key, residual)
            if learn and cache is not None:
                cache.add(key, states.next_ids[t : t + 1])

    return DraftLog(greedy=greedy, alpha=alpha, shallow_nll=shallow_nll)


def expected_speedup(acceptance: float, k: int, n_layers: int, head: float, max_gamma: int = 8) -> tuple[float, int]:
    """Melhor speedup estimado sobre o modelo completo e o γ que o atinge."""
    c_draft = (k + head) / (n_layers + head)
    best = (0.0, 0)
    for gamma in range(1, max_gamma + 1):
        tokens = float(gamma + 1) if acceptance >= 1.0 else (1 - acceptance ** (gamma + 1)) / (1 - acceptance)
        best = max(best, (tokens / (gamma * c_draft + 1.0), gamma))
    return best
