"""Teste matemático local (CPU) da Sinapse Rápida do MYELIN.

Usa um "LLM de brinquedo" congelado, dividido na camada rasa k e no córtex
profundo (camadas k+1..L), e um fluxo de tokens com conceitos recorrentes.
A Camada Rasa começa sem saber nada (estresse alto, acorda o córtex sempre) e,
pelo resíduo da Camada L, aprende o atalho até resolver sozinha.

    pytest -q                          # roda as asserções
    python tests/test_fast_synapse.py  # mostra a Curva de Mielinização
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myelin import FastSynapse, StressGate  # noqa: E402

D, VOCAB, RANK = 64, 32, 16
K_LAYER, L_LAYERS = 4, 28  # profundidade usada no relatório (ex.: Qwen2.5-1.5B tem 28)
TAU = 0.3
ETA = 0.2


def norm(h: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(h, (h.shape[-1],))


class ToyFrozenLM:
    """Substituto de um LLM pré-treinado congelado, cortado na camada rasa k.

    - f_k (shallow): bloco residual raso, aleatório e congelado.
    - córtex (deep): MLP residual seguido de uma memória chave-valor com
      `n_facts` fatos "pré-treinados" (leitura das FFNs como memórias
      chave-valor, Geva et al. 2021). Cada fato liga um conceito a um token.
      O córtex "sabe a resposta": seus estados finais são nítidos (baixa
      entropia), enquanto os da camada rasa são difusos (alta entropia).
    - head: cabeça de leitura compartilhada W_head · Norm(h).
    """

    def __init__(
        self, seed: int = 0, n_facts: int = 64, head_scale: float = 0.15, beta: float = 20.0
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.W_k = torch.randn(D, D, generator=g) / math.sqrt(D)
        self.W_deep1 = torch.randn(D, D, generator=g) / math.sqrt(D)
        self.W_deep2 = torch.randn(D, D, generator=g) / math.sqrt(D)
        self.W_head = torch.randn(VOCAB, D, generator=g) * head_scale
        self.beta = beta

        self.concepts = torch.randn(n_facts, D, generator=g)  # entradas h^(k-1) conhecidas
        self.answers = torch.randint(VOCAB, (n_facts,), generator=g)
        self.fact_keys = norm(self._mlp(self.shallow(self.concepts)))
        self.fact_values = self.W_head[self.answers]

    def shallow(self, h_prev: torch.Tensor) -> torch.Tensor:
        return h_prev + torch.tanh(h_prev @ self.W_k.T)

    def _mlp(self, h_k: torch.Tensor) -> torch.Tensor:
        return h_k + torch.tanh(torch.tanh(h_k @ self.W_deep1.T) @ self.W_deep2.T)

    def deep(self, h_k: torch.Tensor) -> torch.Tensor:
        similarity = norm(self._mlp(h_k)) @ self.fact_keys.T / D  # cosseno
        return F.softmax(self.beta * similarity, dim=-1) @ self.fact_values

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return norm(h) @ self.W_head.T


def concept_stream(
    lm: ToyFrozenLM, n_tokens: int, topic: range = range(8), noise: float = 0.3, seed: int = 0
) -> torch.Tensor:
    """Tokens h^(k-1) sorteados de poucos conceitos recorrentes de um assunto, com ruído."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor(topic)[torch.randint(len(topic), (n_tokens,), generator=g)]
    return lm.concepts[ids] + noise * torch.randn(n_tokens, D, generator=g)


@dataclass
class StreamLog:
    woke: list[bool] = field(default_factory=list)  # o córtex precisou acordar?
    agree: list[bool] = field(default_factory=list)  # camada rasa == camada L?
    stress: list[float] = field(default_factory=list)

    def blocks(self, size: int) -> list[dict[str, float]]:
        out = []
        for start in range(0, len(self.woke), size):
            woke = torch.tensor(self.woke[start : start + size], dtype=torch.float)
            agree = torch.tensor(self.agree[start : start + size], dtype=torch.float)
            # Saída do MYELIN: córtex quando acordou, camada rasa quando não.
            correct = torch.maximum(woke, agree)
            out.append(
                {
                    "deep_rate": woke.mean().item(),
                    "depth": (K_LAYER + woke * (L_LAYERS - K_LAYER)).mean().item(),
                    "accuracy": correct.mean().item(),
                    "shallow_accuracy": agree.mean().item(),
                }
            )
        return out


def run_stream(
    lm: ToyFrozenLM,
    synapse: FastSynapse,
    gate: StressGate,
    tokens: torch.Tensor,
    learn: bool = True,
) -> StreamLog:
    """Laço do MYELIN token a token (online, como numa conversa)."""
    log = StreamLog()
    for h_prev in tokens:
        h_base = lm.shallow(h_prev)  # f_k congelado (modelo de fábrica)
        h_hat = h_base + synapse(h_prev)  # + Sinapse Rápida ΔW_k
        logits_k = lm.head(h_hat)
        stress, wake = gate(logits_k)

        # O córtex roda sobre o estado de fábrica, então o professor é o modelo
        # original, sem realimentação da sinapse. Aqui ele roda sempre só para
        # medir a concordância; no sistema real, só quando wake=True.
        h_L = lm.deep(h_base)
        if learn and wake:
            synapse.consolidate(h_prev, norm(h_L) - norm(h_hat))  # Mielinização

        log.woke.append(bool(wake))
        log.agree.append(bool(logits_k.argmax() == lm.head(h_L).argmax()))
        log.stress.append(float(stress))
    return log


def make_synapse(**kwargs) -> FastSynapse:
    kwargs.setdefault("eta", ETA)
    return FastSynapse(D, D, rank=RANK, **kwargs)


# --------------------------------------------------------------------------- #
# Propriedades matemáticas da sinapse
# --------------------------------------------------------------------------- #


def test_inicio_identico_ao_modelo_de_fabrica():
    lm, synapse = ToyFrozenLM(), make_synapse()
    tokens = concept_stream(lm, 32)
    h_base = lm.shallow(tokens)
    assert torch.equal(h_base + synapse(tokens), h_base)


def test_memoria_fixa_nao_cresce_com_o_uso():
    synapse = make_synapse()
    before = synapse.memory_bytes()
    for _ in range(1000):
        synapse.consolidate(torch.randn(D), torch.randn(D))
    assert synapse.memory_bytes() == before
    assert synapse.B.shape == (D, RANK)


def test_sinapse_fora_do_autograd():
    synapse = make_synapse()
    h = torch.randn(4, D, requires_grad=True)
    synapse.consolidate(h, torch.randn(4, D))
    assert not synapse.A.requires_grad and not synapse.B.requires_grad
    assert list(synapse.parameters()) == []


def test_regra_delta_normalizada_converge_geometricamente():
    """Com γ=1 e NLMS, r_{n+1} = (1-η) r_n exatamente."""
    eta = 0.5
    synapse = make_synapse(eta=eta, gamma=1.0)
    h, target = torch.randn(D), torch.randn(D)
    r0 = target.norm()
    for n in range(1, 11):
        synapse.consolidate(h, target - synapse(h))
        expected = (1 - eta) ** n * r0
        assert torch.isclose((target - synapse(h)).norm(), expected, rtol=1e-3)


def test_mascara_so_aprende_com_tokens_estressados():
    synapse = make_synapse()
    synapse.consolidate(torch.randn(3, D), torch.randn(3, D), mask=torch.zeros(3, dtype=torch.bool))
    assert torch.count_nonzero(synapse.B) == 0


def test_homeostase_limita_a_norma():
    """Com o mesmo estímulo para sempre: γ<1 satura em η·|r|/(|z|(1-γ)); γ=1 explode."""
    h, r = torch.randn(D), torch.randn(D)
    elastic, rigid = make_synapse(gamma=0.99), make_synapse(gamma=1.0)
    for _ in range(3000):
        elastic.consolidate(h, r)
        rigid.consolidate(h, r)
    z_norm = (h @ elastic.A).norm()
    bound = elastic.eta * r.norm() / (z_norm * (1 - elastic.gamma))
    assert elastic.B.norm() <= bound * 1.001
    assert rigid.B.norm() > 10 * elastic.B.norm()


# --------------------------------------------------------------------------- #
# A Curva de Mielinização (versão de brinquedo das hipóteses H1, H2, H3)
# --------------------------------------------------------------------------- #

BLOCK = 40


def test_h1_camada_rasa_aprende_o_atalho():
    lm, synapse, gate = ToyFrozenLM(), make_synapse(), StressGate(tau=TAU)
    blocks = run_stream(lm, synapse, gate, concept_stream(lm, 480)).blocks(BLOCK)

    assert blocks[0]["deep_rate"] > 0.80  # no começo o córtex acorda quase sempre
    assert blocks[-1]["deep_rate"] < 0.35  # no fim a camada rasa resolve sozinha
    assert blocks[-1]["depth"] < 0.5 * blocks[0]["depth"]  # > 2x menos camadas
    assert min(b["accuracy"] for b in blocks) > 0.95  # sem perder a resposta da Camada L


def test_h2_supera_early_exit_estatico():
    lm, gate = ToyFrozenLM(), StressGate(tau=TAU)
    tokens = concept_stream(lm, 480)
    myelin = run_stream(lm, make_synapse(), gate, tokens, learn=True).blocks(BLOCK)
    static = run_stream(lm, make_synapse(), gate, tokens, learn=False).blocks(BLOCK)

    # Sinapses congeladas: a camada rasa nunca fica confiante, o córtex acorda sempre.
    assert static[-1]["deep_rate"] > 0.95
    # E se forçássemos a saída rasa, ela erraria: o atalho só existe com a sinapse viva.
    assert myelin[-1]["shallow_accuracy"] > static[-1]["shallow_accuracy"] + 0.5


def test_h3_readapta_quando_o_assunto_muda():
    lm, synapse, gate = ToyFrozenLM(), make_synapse(), StressGate(tau=TAU)
    topic_a = concept_stream(lm, 400, topic=range(0, 8), seed=1)
    topic_b = concept_stream(lm, 400, topic=range(8, 16), seed=2)
    blocks = run_stream(lm, synapse, gate, torch.cat([topic_a, topic_b])).blocks(BLOCK)
    switch = len(topic_a) // BLOCK

    assert blocks[switch - 1]["deep_rate"] < 0.35  # assunto A consolidado
    assert blocks[switch]["deep_rate"] > 0.70  # assunto novo: estresse volta
    assert blocks[-1]["deep_rate"] < 0.35  # assunto B consolidado
    assert min(b["accuracy"] for b in blocks) > 0.95


# --------------------------------------------------------------------------- #
# Demonstração: python tests/test_fast_synapse.py
# --------------------------------------------------------------------------- #


def _print_curve(title: str, blocks: list[dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'tokens':>9} | {'córtex':>6} | {'camadas/token':>13} | {'acurácia':>8} | profundidade")
    for i, b in enumerate(blocks):
        bar = "█" * round(b["depth"]) + "·" * (L_LAYERS - round(b["depth"]))
        print(
            f"{i * BLOCK:>4}-{(i + 1) * BLOCK - 1:<4} | {b['deep_rate']:>6.0%} | "
            f"{b['depth']:>13.1f} | {b['accuracy']:>8.0%} | {bar}"
        )


def main() -> None:
    lm, gate = ToyFrozenLM(), StressGate(tau=TAU)
    tokens = concept_stream(lm, 480)

    synapse = make_synapse()
    log = run_stream(lm, synapse, gate, tokens)
    _print_curve(
        f"MYELIN — camada rasa k={K_LAYER}, córtex até L={L_LAYERS}, "
        f"sinapse r={RANK} ({synapse.memory_bytes() / 1024:.0f} KiB fixos), η={ETA}, τ={TAU}",
        log.blocks(BLOCK),
    )

    static = run_stream(lm, make_synapse(), gate, tokens, learn=False)
    _print_curve("Early-Exit Estático (sinapse congelada, ΔW_k = 0)", static.blocks(BLOCK))

    shift = make_synapse()
    both = torch.cat(
        [
            concept_stream(lm, 400, topic=range(0, 8), seed=1),
            concept_stream(lm, 400, topic=range(8, 16), seed=2),
        ]
    )
    _print_curve(
        "Mudança de assunto no token 400 (homeostase γ=0.995)",
        run_stream(lm, shift, gate, both).blocks(BLOCK),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
