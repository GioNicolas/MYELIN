"""Acoplador do MYELIN a modelos HuggingFace (Qwen / Llama), em modo simulação.

O modelo roda inteiro, uma vez por texto, e o portão só decide qual previsão
*contaria*: a da camada rasa k (saída cedo) ou a da camada final L (córtex).
O córtex sempre trabalha sobre o estado de fábrica f_k(h), sem a sinapse.

Limites da simulação (não é a economia real):
  - ignora o KV cache das camadas k+1..L, que numa saída cedo de verdade
    ficaria faltando para os tokens seguintes (recalcular custa caro;
    aproximar muda o professor);
  - não conta o custo da leitura rasa: no Qwen2.5-0.5B a cabeça (d × |V|,
    ~136 M de pesos) custa mais que as 7 camadas que uma saída em k = 17 pula.

Leitura da camada rasa, com a Sinapse Rápida "em cima da tradução" P_k:

    s_t   = P_k(h_t^(k))
    q_t   = s_t + B (A^T key_t)                    key_t: ver make_keys
    p_t^k = Softmax(W_head · Norm(q_t))

Mielinização, quando o córtex acordou: o alvo é a direção do estado final
(antes do Norm) na escala de s_t, uma escala fixa por token que não depende
da sinapse (escalar pelo próprio q_t realimentaria a correção):

    r_t = rms(s_t) · h_t^(L) / rms(h_t^(L)) - q_t

Com η = 1, γ = 1 e ε → 0, um passo faz q_t = alvo e a camada rasa passa a dar
a distribuição final para aquela mesma chave.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Protocol

import torch
import torch.nn.functional as F

from myelin.lens import TunedLens
from myelin.stress_gate import StressGate

Readout = Callable[[torch.Tensor], torch.Tensor]


class Memory(Protocol):
    """O que a simulação usa de uma memória rápida (FastSynapse ou KNNCache)."""

    def __call__(self, key: torch.Tensor) -> torch.Tensor: ...
    def consolidate(self, h_in: torch.Tensor, residual: torch.Tensor, mask=None, decay: bool = True) -> None: ...
    def decay(self) -> None: ...


def rms_normalize(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """h / rms(h): mesma direção, escala 1 por coordenada (o Norm sem o peso γ)."""
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def causal_center(h: torch.Tensor) -> torch.Tensor:
    """Subtrai de cada h_t a média de h_0..h_{t-1} (h_0 fica como está).

    Cuidado: com dimensões de ativação gigante que aparecem só em alguns
    tokens (no Qwen, o primeiro e alguns delimitadores), TODO token passa a
    desviar muito da média nessas dimensões e as chaves ficam quase paralelas.
    Foi a causa provável do colapso na rodada 1 do Experimento 1.
    """
    counts = torch.arange(1, h.shape[0], device=h.device, dtype=h.dtype).unsqueeze(-1)
    past_mean = h[:-1].cumsum(0) / counts
    return torch.cat([h[:1], h[1:] - past_mean])


def causal_standardize(h: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Padroniza cada dimensão pela média e desvio de h_0..h_t (inclui o próprio t).

    Cuidado: com poucos tokens a estimativa é degenerada. O token 0 vira o
    vetor nulo, e no token 1 toda dimensão vira ±1 (com 2 amostras, padronizar
    só deixa o sinal da diferença). Prefira estatísticas fixas (`fit_key_stats`).
    """
    h64 = h.double()
    counts = torch.arange(1, h.shape[0] + 1, device=h.device, dtype=torch.float64).unsqueeze(-1)
    mean = h64.cumsum(0) / counts
    var = (h64.pow(2).cumsum(0) / counts - mean.pow(2)).clamp_min(0.0)
    return ((h64 - mean) / (var + eps).sqrt()).to(h.dtype)


@dataclass
class KeyStats:
    """Média e desvio por dimensão de um estado, medidos fora do texto avaliado."""

    mean: torch.Tensor
    std: torch.Tensor


KEY_MODES = ("raw", "center", "standardize", "fixed", "bias")


def make_keys(h: torch.Tensor, mode: str = "fixed", stats: KeyStats | None = None) -> torch.Tensor:
    """Chaves da sinapse a partir de um estado cru (h^(k-1) ou h^(k)), normalizadas por RMS.

    raw: só normaliza; center: subtrai a média causal; standardize: padroniza
    cada dimensão de forma causal; fixed: padroniza com estatísticas fixas
    medidas num corpus separado; bias: a mesma chave para todo token, ou seja,
    a "sinapse" vira um viés aprendido ao vivo, sem olhar o contexto (ablação:
    a geometria das chaves importa, ou só se corrige um erro global do P_k?).
    """
    if mode == "raw":
        return rms_normalize(h)
    if mode == "center":
        return rms_normalize(causal_center(h))
    if mode == "standardize":
        return rms_normalize(causal_standardize(h))
    if mode == "fixed":
        if stats is None:
            raise ValueError("o modo 'fixed' precisa de stats (ver fit_key_stats)")
        return rms_normalize((h - stats.mean) / stats.std)
    if mode == "bias":
        return torch.ones_like(h)
    raise ValueError(f"modo de chave desconhecido {mode!r}; use {KEY_MODES}")


def key_interference(
    key: torch.Tensor, A: torch.Tensor | None = None, n_pairs: int = 4000, skip: int = 8, seed: int = 0
) -> dict[str, float]:
    """Quanto uma consolidação num token mexe na leitura de outro.

    Com z = A^T key, ensinar o token t muda a correção do token j por
    η · r_t · c_tj, com c_tj = z_t·z_j / ||z_t||². Devolve os quantis de |c_tj|
    em pares sorteados (sem os `skip` primeiros tokens). Perto de 0: cada
    consolidação fica no seu token; perto de 1: vale para o texto todo.
    """
    z = key[skip:].float()
    if A is not None:
        z = z @ A.to(z)
    g = torch.Generator().manual_seed(seed)
    i, j = torch.randint(len(z), (2, n_pairs), generator=g)
    i, j = i[i != j], j[i != j]
    c = ((z[i] * z[j]).sum(-1) / (z[i].pow(2).sum(-1) + 1e-6)).abs()
    q = torch.quantile(c, torch.tensor([0.5, 0.9, 0.99])).tolist()
    return {"p50": q[0], "p90": q[1], "p99": q[2]}


@dataclass
class LayerStates:
    """O que a simulação precisa de um texto, para uma camada rasa k.

    Atributos (T = número de tokens, d = dimensão do modelo):
        h_in: entrada crua da camada k, h^(k-1), (T, d).
        h_out: saída crua da camada k, h^(k), (T, d).
        key: chaves da memória rápida (ver `make_keys`), (T, d).
        shallow: estado raso traduzido P_k(h^(k)), sem a sinapse, (T, d).
        target: direção do estado final, h^(L) / rms(h^(L)), antes do Norm, (T, d).
        final_top: top-1 da camada final, (T,).
        final_nll: -log p_L(próximo token real), (T,); NaN no último token.
        next_ids: próximo token real, (T,); -1 no último token.
        k, n_layers: camada rasa e total de camadas do modelo.
    """

    h_in: torch.Tensor
    h_out: torch.Tensor
    key: torch.Tensor
    shallow: torch.Tensor
    target: torch.Tensor
    final_top: torch.Tensor
    final_nll: torch.Tensor
    next_ids: torch.Tensor
    k: int
    n_layers: int


def hf_readout(model) -> Readout:
    """Cabeça do próprio modelo: q -> lm_head(norm(q))."""
    return lambda q: model.lm_head(model.model.norm(q))


@torch.no_grad()
def fit_key_stats(model, texts_ids: Sequence[torch.Tensor], k: int, skip: int = 4) -> dict[str, KeyStats]:
    """Média e desvio de h^(k-1) ("input") e h^(k) ("output") num corpus de calibração.

    Os `skip` primeiros tokens de cada texto ficam de fora: o primeiro token
    concentra as ativações gigantes (attention sink) e distorceria o desvio.
    """
    states = {"input": [], "output": []}
    for ids in texts_ids:
        hidden = model(ids, output_hidden_states=True).hidden_states
        states["input"].append(hidden[k - 1][0, skip:].float())
        states["output"].append(hidden[k][0, skip:].float())
    stats = {}
    for name, chunks in states.items():
        h = torch.cat(chunks)
        stats[name] = KeyStats(mean=h.mean(0), std=h.std(0).clamp_min(1e-3))
    return stats


@torch.no_grad()
def capture(
    model,
    input_ids: torch.Tensor,
    k: int,
    lens: TunedLens | None = None,
    key_mode: str = "raw",
    stats: KeyStats | None = None,
) -> LayerStates:
    """Roda o modelo congelado uma vez e guarda só o que a camada k precisa.

    Args:
        model: *CausalLM do HuggingFace com `model.model.norm` e `model.lm_head`.
        input_ids: (1, T).
        k: camada rasa, 1..L-1.
        lens: tradutor P_k; None = logit lens (P_k = identidade).
        key_mode, stats: chaves feitas de h^(k-1) (ver `make_keys`). Para
            outras chaves, troque `key` depois com `dataclasses.replace`.
    """
    n_layers = model.config.num_hidden_layers
    if not 1 <= k < n_layers:
        raise ValueError(f"k deve estar em 1..{n_layers - 1}, recebido {k}")

    captured = {}
    hook = model.model.norm.register_forward_hook(lambda m, inp, out: captured.update(h_final=inp[0]))
    try:
        out = model(input_ids, output_hidden_states=True)
    finally:
        hook.remove()

    # hidden_states[0] é o embedding; [ℓ] é a saída da camada ℓ (= entrada da ℓ+1).
    h_in, h_k = out.hidden_states[k - 1][0].float(), out.hidden_states[k][0].float()
    shallow = h_k if lens is None else lens(k, h_k)

    logits = out.logits[0].float()
    next_ids = torch.cat([input_ids[0, 1:], input_ids.new_full((1,), -1)])
    final_nll = torch.full((len(next_ids),), float("nan"), device=logits.device)
    final_nll[:-1] = F.cross_entropy(logits[:-1], next_ids[:-1], reduction="none")

    return LayerStates(
        h_in=h_in,
        h_out=h_k,
        key=make_keys(h_in, key_mode, stats),
        shallow=shallow,
        target=rms_normalize(captured["h_final"][0].float()),
        final_top=logits.argmax(-1),
        final_nll=final_nll,
        next_ids=next_ids,
        k=k,
        n_layers=n_layers,
    )


@dataclass
class SimulationLog:
    """Registro token a token de uma simulação; todos os campos têm forma (T,).

    Toda leitura rasa é registrada ANTES de a memória aprender com aquele
    token, então shallow_agree e shallow_nll medem o que a memória já sabia.
    """

    stress: torch.Tensor  # estresse da camada rasa (com a sinapse daquele instante)
    exit: torch.Tensor  # saiu cedo: a previsão da camada rasa foi a usada
    audit: torch.Tensor  # o córtex acordou por sorteio, mesmo com estresse baixo
    shallow_agree: torch.Tensor  # top-1 raso == top-1 final, saindo ou não
    shallow_nll: torch.Tensor  # -log p_k(próximo token real), saindo ou não; NaN no último
    agree: torch.Tensor  # a previsão usada == top-1 final (sempre True sem saída)
    nll: torch.Tensor  # -log p(próximo token real) da previsão usada; NaN no último
    depth: torch.Tensor  # camadas usadas: k na saída cedo, L com o córtex

    def blocks(self, n_blocks: int) -> list[dict[str, float]]:
        """Resume o texto em `n_blocks` trechos consecutivos: a Curva de Mielinização."""

        def ppl(nll: torch.Tensor) -> float:
            valid = ~nll.isnan()
            return nll[valid].mean().exp().item() if valid.any() else float("nan")

        rows = []
        chunks = zip(*(getattr(self, f.name).float().chunk(n_blocks) for f in fields(self)))
        for i, (stress, exit, audit, shallow_agree, shallow_nll, agree, nll, depth) in enumerate(chunks):
            exits = exit.bool()
            rows.append(
                {
                    "block": i,
                    "tokens": len(exit),
                    "exit_rate": exit.mean().item(),
                    "depth": depth.mean().item(),
                    "agree": agree.mean().item(),
                    "shallow_agree": shallow_agree.mean().item(),
                    "exit_precision": agree[exits].mean().item() if exits.any() else float("nan"),
                    "ppl": ppl(nll),
                    "shallow_ppl": ppl(shallow_nll),
                    "stress": stress.mean().item(),
                }
            )
        return rows


@torch.no_grad()
def simulate(
    states: LayerStates,
    readout: Readout,
    synapse: Memory,
    gate: StressGate,
    learn: bool = True,
    audit_rate: float = 0.0,
    seed: int = 0,
    teacher_every: int = 0,
    decay_per_token: bool = False,
    target_scale: str = "shallow",
    max_correction: float | None = None,
) -> SimulationLog:
    """Percorre o texto token a token, na ordem, como o MYELIN faria ao vivo.

    Para cada token: lê a camada rasa com a sinapse atual, o portão decide, e,
    se o córtex acordou e `learn`, a sinapse consolida o resíduo daquele token
    antes do próximo. Com `learn=False`, é o Early-Exit Estático com o mesmo
    P_k: a linha de base do H2.

    Args:
        audit_rate: probabilidade de acordar o córtex mesmo com estresse baixo.
            Sem isso, um token confiante e errado nunca é corrigido, porque só
            o córtex ensina a sinapse.
        teacher_every: se > 0, ignora o portão e acorda o córtex num calendário
            fixo (tokens 0, n, 2n, ...). Separa "a memória aprende algo útil
            para os tokens seguintes?" de "o portão escolhe bem quando acordar?".
        decay_per_token: aplica γ a cada token, e não só quando há consolidação
            (senão, se o córtex para de acordar, a sinapse para de esquecer).
        target_scale: "shallow" (escala fixa rms(s_t)) ou "q" (rms(q_t), como
            na rodada 1 do Experimento 1).
        max_correction: se dado, limita ||B z|| a esse múltiplo de ||s_t||
            (região de confiança: a sinapse não pode esmagar a tradução).
    """
    if target_scale not in ("shallow", "q"):
        raise ValueError(f"target_scale deve ser 'shallow' ou 'q', recebido {target_scale!r}")
    T = states.key.shape[0]
    g = torch.Generator().manual_seed(seed)
    audit = torch.rand(T, generator=g) < audit_rate

    # Trechos em que a memória não muda, lidos de uma vez (a cabeça d × |V| é o
    # custo dominante). Com o portão, cada token pode ensinar: trechos de 1.
    # Com calendário fixo, o trecho vai até o próximo token do córtex, inclusive:
    # ele é lido antes de ensinar, como os outros.
    # Sem aprender (o estático), a memória nunca muda: lotes de 128 tokens.
    if teacher_every and not decay_per_token:
        starts = [0, *range(1, T, teacher_every), T]
    elif not learn and not decay_per_token:
        starts = [*range(0, T, 128), T]
    else:
        starts = list(range(T + 1))

    log = {name: torch.zeros(T, dtype=torch.bool) for name in ("exit", "audit", "shallow_agree", "agree")}
    stress_log = torch.zeros(T)
    nll_log, shallow_nll_log = torch.full((T,), float("nan")), torch.full((T,), float("nan"))

    def rms(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()

    for a, b in zip(starts[:-1], starts[1:]):
        if decay_per_token:
            synapse.decay()
        key, shallow = states.key[a:b], states.shallow[a:b]
        correction = synapse(key)
        if max_correction is not None:
            limit = max_correction * shallow.norm(dim=-1, keepdim=True)
            correction = correction * (limit / correction.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        q = shallow + correction
        logits = readout(q).float()
        stress, gate_wake = (x.cpu() for x in gate(logits))
        positions = torch.arange(a, b)
        wake = positions % teacher_every == 0 if teacher_every else gate_wake | audit[a:b]
        exit = ~wake

        targets = states.next_ids[a:b]
        log_p = F.log_softmax(logits, -1)
        shallow_nll = -log_p.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1).cpu()
        shallow_nll[targets.cpu() < 0] = float("nan")
        shallow_agree = logits.argmax(-1).cpu() == states.final_top[a:b].cpu()

        stress_log[a:b] = stress
        log["exit"][a:b] = exit
        log["audit"][a:b] = audit[a:b] & ~gate_wake & ~torch.tensor(bool(teacher_every))
        log["shallow_agree"][a:b] = shallow_agree
        log["agree"][a:b] = shallow_agree | wake
        shallow_nll_log[a:b] = shallow_nll
        nll_log[a:b] = torch.where(exit, shallow_nll, states.final_nll[a:b].cpu())

        if learn:
            for j in wake.nonzero().flatten().tolist():  # só o último do trecho pode acordar
                scale = rms(shallow[j : j + 1]) if target_scale == "shallow" else rms(q[j : j + 1])
                residual = scale * states.target[a + j : a + j + 1] - q[j : j + 1]
                synapse.consolidate(key[j : j + 1], residual, decay=not decay_per_token)

    depth = torch.where(log["exit"], states.k, states.n_layers).float()
    return SimulationLog(
        stress=stress_log,
        exit=log["exit"],
        audit=log["audit"],
        shallow_agree=log["shallow_agree"],
        shallow_nll=shallow_nll_log,
        agree=log["agree"],
        nll=nll_log,
        depth=depth,
    )
