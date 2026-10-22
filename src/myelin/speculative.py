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
from typing import Protocol

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


# ---------------------------------------------------------------------------
# Experimento 7: simulação do ciclo real (rascunho → verificação → token bônus)
# ---------------------------------------------------------------------------
#
# Sobre uma sequência que o próprio modelo gerou (gulosa), o ciclo é exato: na
# posição c (o último token emitido), o rascunhador propõe até γ tokens; o
# verificador aceita o maior prefixo igual ao seu top-1 e emite mais um token
# (a correção, ou o bônus se aceitou todos). Só então o rascunhador aprende,
# com as posições verificadas: o professor chega com o atraso de um ciclo.
#
# Custo de um ciclo, em passos do modelo completo (L + H camadas):
#     custo = custo_rascunho / (L + H) + 1 + ε · (tokens rascunhados)
# ε é o custo extra de cada posição a mais na verificação (que é limitada por
# memória, mas não é de graça). Sem proposta, o ciclo é um passo normal (custo 1).


class Drafter(Protocol):
    def propose(self, c: int, gamma: int) -> tuple[list[int], float]:
        """Até γ tokens para as posições c+1.. e o custo do rascunho em camadas."""
        ...

    def observe(self, positions: range) -> None:
        """O verificador acabou de dar o professor destas posições."""
        ...


@dataclass
class CycleStats:
    tokens: int = 0
    cost: float = 0.0  # em passos do modelo completo
    cycles: int = 0
    proposals: int = 0  # ciclos em que houve rascunho
    drafted: int = 0
    accepted: int = 0

    @property
    def speedup(self) -> float:
        return self.tokens / self.cost if self.cost else float("nan")

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "speedup": self.speedup,
            "acceptance": self.acceptance,
            "tokens_per_cycle": self.tokens / self.cycles if self.cycles else float("nan"),
            "proposal_rate": self.proposals / self.cycles if self.cycles else float("nan"),
            "tokens": self.tokens,
        }


def run_cycles(
    drafter: Drafter,
    final_top: list[int],
    start: int,
    end: int,
    gamma: int,
    full_cost: float,
    extra_token_cost: float = 0.02,
) -> CycleStats:
    """Percorre as posições start..end-1; final_top[p] é o token que o verificador emite depois de p."""
    stats, c = CycleStats(), start
    while c < end:
        drafts, draft_cost = drafter.propose(c, gamma)
        accepted = 0
        for i, token in enumerate(drafts):
            if c + i >= end or token != final_top[c + i]:
                break
            accepted += 1
        produced = min(accepted + 1, end - c)
        stats.cost += draft_cost / full_cost + 1.0 + extra_token_cost * len(drafts)
        stats.tokens += produced
        stats.cycles += 1
        stats.proposals += bool(drafts)
        stats.drafted += len(drafts)
        stats.accepted += min(accepted, produced)
        drafter.observe(range(c, c + produced))
        c += produced
    return stats


class HeadDrafter:
    """Rascunho pela camada k: lm_head(norm(P_k(h_k) + memória)), pagando k + H camadas por token.

    A memória (ou o cache neural) fica fixa durante o ciclo e aprende em
    `observe`, com o estado final das posições verificadas.
    """

    def __init__(
        self,
        states: LayerStates,
        readout: Readout,
        head: float,
        memory: Memory | None = None,
        cache: NeuralCache | None = None,
    ) -> None:
        self.states, self.readout, self.memory, self.cache = states, readout, memory, cache
        self.final_top = states.final_top.tolist()
        self.step_cost = states.k + head

    def _logits(self, p: int) -> torch.Tensor:
        key, q = self.states.key[p : p + 1], self.states.shallow[p : p + 1]
        if self.memory is not None:
            q = q + self.memory(key)
        logits = self.readout(q).float()
        return self.cache.mix(key, logits) if self.cache is not None else logits

    @torch.no_grad()
    def propose(self, c: int, gamma: int) -> tuple[list[int], float]:
        drafts: list[int] = []
        for p in range(c, min(c + gamma, len(self.final_top))):
            if drafts and drafts[-1] != self.final_top[p - 1]:
                drafts.append(-1)  # o prefixo já divergiu: este rascunho será descartado (mas foi pago)
                continue
            drafts.append(int(self._logits(p).argmax(-1)))
        return drafts, gamma * self.step_cost

    @torch.no_grad()
    def observe(self, positions: range) -> None:
        for p in positions:
            key, shallow = self.states.key[p : p + 1], self.states.shallow[p : p + 1]
            if self.memory is not None:
                q = shallow + self.memory(key)
                scale = shallow.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                self.memory.consolidate(key, scale * self.states.target[p : p + 1] - q)
            if self.cache is not None:
                self.cache.add(key, self.states.final_top[p : p + 1])


class PromptLookupDrafter:
    """Linha de base sem camada nenhuma: o último n-grama emitido já apareceu? Propõe o que veio depois.

    É o Prompt Lookup Decoding (Saxena, 2023), com recuo de n_max até n_min.
    """

    def __init__(self, tokens: list[int], n_max: int = 3, n_min: int = 1) -> None:
        self.tokens, self.n_max, self.n_min = tokens, n_max, n_min
        self.table: dict[int, dict[tuple[int, ...], int]] = {n: {} for n in range(n_min, n_max + 1)}
        self.indexed = -1

    def propose(self, c: int, gamma: int) -> tuple[list[int], float]:
        # Indexa os n-gramas que terminam até c-1: o que veio depois deles já é conhecido.
        while self.indexed < c - 1:
            self.indexed += 1
            for n in self.table:
                if self.indexed - n + 1 >= 0:
                    self.table[n][tuple(self.tokens[self.indexed - n + 1 : self.indexed + 1])] = self.indexed
        for n in range(self.n_max, self.n_min - 1, -1):
            if c - n + 1 < 0:
                continue
            j = self.table[n].get(tuple(self.tokens[c - n + 1 : c + 1]))
            if j is not None:
                return self.tokens[j + 1 : min(j + 1 + gamma, c + 1)], 0.0
        return [], 0.0

    def observe(self, positions: range) -> None:
        pass


class ReflexDrafter:
    """O reflexo do MYELIN: reconhecer a situação pelo estado oculto e repetir o que o verificador disse.

    Chave: o estado da camada l na posição c-1 (já calculado no ciclo anterior,
    então de graça). Procura, entre as últimas `capacity` posições i, a mais
    parecida (cos >= tau) cujo token seguinte é o que acabou de ser emitido
    (tokens[i+1] == tokens[c]) e propõe a continuação tokens[i+2..]. Sem
    vizinho familiar, não especula: nenhum custo a mais.
    """

    def __init__(self, keys: torch.Tensor, tokens: list[int], tau: float = 0.8, capacity: int = 1024) -> None:
        self.keys = F.normalize(keys.float(), dim=-1)
        self.tokens, self.tau, self.capacity = tokens, tau, capacity
        self.ids = torch.tensor(tokens, device=keys.device)

    @torch.no_grad()
    def propose(self, c: int, gamma: int) -> tuple[list[int], float]:
        if c < 2:
            return [], 0.0
        lo = max(0, c - 1 - self.capacity)
        sims = self.keys[lo : c - 1] @ self.keys[c - 1]  # candidatos i = lo..c-2
        consistent = self.ids[lo + 1 : c] == self.tokens[c]
        sims = torch.where(consistent & (sims >= self.tau), sims, torch.full_like(sims, -2.0))
        best = int(sims.argmax())
        if sims[best] < -1.0:
            return [], 0.0
        i = lo + best
        return self.tokens[i + 2 : min(i + 2 + gamma, c + 1)], 0.0

    def observe(self, positions: range) -> None:
        pass


class ChainDrafter:
    """Tenta o primeiro rascunhador; se ele não propuser nada, o segundo."""

    def __init__(self, first: Drafter, second: Drafter) -> None:
        self.first, self.second = first, second

    def propose(self, c: int, gamma: int) -> tuple[list[int], float]:
        drafts, cost = self.first.propose(c, gamma)
        if drafts:
            return drafts, cost
        more, more_cost = self.second.propose(c, gamma)
        return more, cost + more_cost

    def observe(self, positions: range) -> None:
        self.first.observe(positions)
        self.second.observe(positions)
