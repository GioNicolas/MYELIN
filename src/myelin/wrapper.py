"""Acoplador do MYELIN a modelos HuggingFace (Qwen / Llama), em modo simulação.

O modelo roda inteiro, uma vez por texto, e o portão só decide qual previsão
*contaria*: a da camada rasa k (saída cedo) ou a da camada final L (córtex).
Como o córtex sempre trabalha sobre o estado de fábrica f_k(h), sem a sinapse,
as previsões simuladas são exatamente as que o MYELIN daria. A única coisa
que a simulação ignora é o KV cache das camadas k+1..L, que numa saída cedo
de verdade ficaria faltando para os tokens seguintes.

Leitura da camada rasa, com a Sinapse Rápida "em cima da tradução" P_k:

    q_t   = P_k(h_t^(k)) + B (A^T key_t)          key_t = h_t^(k-1) centrado e normalizado
    p_t^k = Softmax(W_head · Norm(q_t))

Mielinização, quando o córtex acordou: o alvo é a direção do estado final
(antes do Norm) na escala de q_t, de modo que um passo com η = 1 faz
Norm(q_t) = Norm(h_t^(L)) e a camada rasa passa a dar a distribuição final:

    r_t = rms(q_t) · h_t^(L) / rms(h_t^(L)) - q_t
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from myelin.fast_synapse import FastSynapse
from myelin.lens import TunedLens
from myelin.stress_gate import StressGate

Readout = Callable[[torch.Tensor], torch.Tensor]


def rms_normalize(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """h / rms(h): mesma direção, escala 1 por coordenada (o Norm sem o peso γ)."""
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def causal_center(h: torch.Tensor) -> torch.Tensor:
    """Subtrai de cada h_t a média de h_0..h_{t-1} (h_0 fica como está).

    Os estados do Qwen têm uma componente comum enorme (dimensões com
    ativações gigantes), que deixaria todas as chaves A^T h quase paralelas e
    faria cada consolidação interferir em todos os tokens. A média usa só o
    passado, então não espia o resto do texto.
    """
    counts = torch.arange(1, h.shape[0], device=h.device, dtype=h.dtype).unsqueeze(-1)
    past_mean = h[:-1].cumsum(0) / counts
    return torch.cat([h[:1], h[1:] - past_mean])


@dataclass
class LayerStates:
    """O que a simulação precisa de um texto, para uma camada rasa k.

    Atributos (T = número de tokens, d = dimensão do modelo):
        key: chaves da sinapse, h^(k-1) centrado e normalizado, (T, d).
        shallow: estado raso traduzido P_k(h^(k)), sem a sinapse, (T, d).
        target: direção do estado final, h^(L) / rms(h^(L)), antes do Norm, (T, d).
        final_top: top-1 da camada final, (T,).
        final_nll: -log p_L(próximo token real), (T,); NaN no último token.
        next_ids: próximo token real, (T,); -1 no último token.
        k, n_layers: camada rasa e total de camadas do modelo.
    """

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
def capture(
    model, input_ids: torch.Tensor, k: int, lens: TunedLens | None = None, center_keys: bool = True
) -> LayerStates:
    """Roda o modelo congelado uma vez e guarda só o que a camada k precisa.

    Args:
        model: *CausalLM do HuggingFace com `model.model.norm` e `model.lm_head`.
        input_ids: (1, T).
        k: camada rasa, 1..L-1.
        lens: tradutor P_k; None = logit lens (P_k = identidade).
        center_keys: centra as chaves pela média causal (ver `causal_center`).
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
    key = rms_normalize(causal_center(h_in) if center_keys else h_in)
    shallow = h_k if lens is None else lens(k, h_k)

    logits = out.logits[0].float()
    next_ids = torch.cat([input_ids[0, 1:], input_ids.new_full((1,), -1)])
    final_nll = torch.full((len(next_ids),), float("nan"), device=logits.device)
    final_nll[:-1] = F.cross_entropy(logits[:-1], next_ids[:-1], reduction="none")

    return LayerStates(
        key=key,
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
    """Registro token a token de uma simulação; todos os campos têm forma (T,)."""

    stress: torch.Tensor  # estresse da camada rasa (com a sinapse daquele instante)
    exit: torch.Tensor  # saiu cedo: a previsão da camada rasa foi a usada
    audit: torch.Tensor  # o córtex acordou por sorteio, mesmo com estresse baixo
    shallow_agree: torch.Tensor  # top-1 raso == top-1 final, saindo ou não
    agree: torch.Tensor  # a previsão usada == top-1 final (sempre True sem saída)
    nll: torch.Tensor  # -log p(próximo token real) da previsão usada; NaN no último
    depth: torch.Tensor  # camadas usadas: k na saída cedo, L com o córtex

    def blocks(self, n_blocks: int) -> list[dict[str, float]]:
        """Resume o texto em `n_blocks` trechos consecutivos: a Curva de Mielinização."""
        rows = []
        chunks = zip(*(getattr(self, f.name).float().chunk(n_blocks) for f in fields(self)))
        for i, (stress, exit, audit, shallow_agree, agree, nll, depth) in enumerate(chunks):
            exits = exit.bool()
            valid = ~nll.isnan()
            rows.append(
                {
                    "block": i,
                    "tokens": len(exit),
                    "exit_rate": exit.mean().item(),
                    "depth": depth.mean().item(),
                    "agree": agree.mean().item(),
                    "shallow_agree": shallow_agree.mean().item(),
                    "exit_precision": agree[exits].mean().item() if exits.any() else float("nan"),
                    "ppl": nll[valid].mean().exp().item() if valid.any() else float("nan"),
                    "stress": stress.mean().item(),
                }
            )
        return rows


@torch.no_grad()
def simulate(
    states: LayerStates,
    readout: Readout,
    synapse: FastSynapse,
    gate: StressGate,
    learn: bool = True,
    audit_rate: float = 0.0,
    seed: int = 0,
) -> SimulationLog:
    """Percorre o texto token a token, na ordem, como o MYELIN faria ao vivo.

    Para cada token: lê a camada rasa com a sinapse atual, o portão decide, e,
    se o córtex acordou e `learn`, a sinapse consolida o resíduo daquele token
    antes do próximo. Com `learn=False` (ou a sinapse zerada e sem aprender),
    é o Early-Exit Estático com o mesmo P_k: a linha de base do H2.

    Args:
        audit_rate: probabilidade de acordar o córtex mesmo com estresse baixo.
            Sem isso, um token confiante e errado nunca é corrigido, porque só
            o córtex ensina a sinapse.
    """
    T = states.key.shape[0]
    g = torch.Generator().manual_seed(seed)
    audit = torch.rand(T, generator=g) < audit_rate

    log = {name: torch.zeros(T, dtype=torch.bool) for name in ("exit", "audit", "shallow_agree", "agree")}
    stress_log, nll_log = torch.zeros(T), torch.full((T,), float("nan"))

    for t in range(T):
        key = states.key[t : t + 1]
        q = states.shallow[t : t + 1] + synapse(key)
        logits = readout(q).float()
        stress, wake = gate(logits)
        exit = not wake.item() and not audit[t].item()
        shallow_top = logits.argmax(-1).item()
        final_top = states.final_top[t].item()
        target = states.next_ids[t].item()

        stress_log[t] = stress.item()
        log["exit"][t] = exit
        log["audit"][t] = audit[t].item() and not wake.item()
        log["shallow_agree"][t] = shallow_top == final_top
        if exit:
            log["agree"][t] = shallow_top == final_top
            if target >= 0:
                nll_log[t] = -F.log_softmax(logits[0], -1)[target].item()
        else:
            log["agree"][t] = True
            nll_log[t] = states.final_nll[t].item()
            if learn:
                scale = q.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                residual = scale * states.target[t : t + 1] - q
                synapse.consolidate(key, residual)

    depth = torch.where(log["exit"], states.k, states.n_layers).float()
    return SimulationLog(
        stress=stress_log,
        exit=log["exit"],
        audit=log["audit"],
        shallow_agree=log["shallow_agree"],
        agree=log["agree"],
        nll=nll_log,
        depth=depth,
    )
