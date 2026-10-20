"""Testes do acoplador em modo simulação.

Um Qwen2 minúsculo e aleatório testa a captura dos estados (roda em CPU em
menos de 1 s); estados sintéticos com padrões recorrentes testam a lógica do
portão e da Mielinização sem depender de um modelo treinado.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from myelin import FastSynapse, StressGate, TunedLens
from myelin.wrapper import (
    LayerStates,
    capture,
    causal_center,
    causal_standardize,
    hf_readout,
    key_similarity,
    make_keys,
    rms_normalize,
    simulate,
)

D, VOCAB = 64, 64


@pytest.fixture(scope="module")
def tiny_qwen():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def test_centro_causal_usa_so_o_passado():
    h = torch.randn(6, 4)
    centered = causal_center(h)
    assert torch.equal(centered[0], h[0])
    for t in range(1, 6):
        torch.testing.assert_close(centered[t], h[t] - h[:t].mean(0))


def test_padronizacao_causal_usa_so_ate_t():
    h = torch.randn(6, 4) * torch.tensor([1.0, 10.0, 100.0, 1000.0])
    standardized = causal_standardize(h)
    assert torch.equal(standardized[0], torch.zeros(4))
    for t in range(1, 6):
        past = h[: t + 1].double()
        expected = (past[-1] - past.mean(0)) / (past.var(0, unbiased=False) + 1e-5).sqrt()
        torch.testing.assert_close(standardized[t], expected.float())


def test_padronizar_tira_o_dominio_de_uma_dimensao_gigante():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(200, 32, generator=g)
    h[:, 0] += 1000.0 * (torch.rand(200, generator=g) < 0.3)  # ativação gigante em 30% dos tokens
    # Centrar piora: todo token passa a desviar muito da média nessa dimensão,
    # e todas as chaves ficam quase paralelas. Padronizar resolve.
    assert key_similarity(make_keys(h, "center")) > 0.9
    assert key_similarity(make_keys(h, "standardize")) < 0.2 < key_similarity(make_keys(h, "raw"))


def test_modo_de_chave_desconhecido():
    with pytest.raises(ValueError):
        make_keys(torch.randn(3, 4), "whiten")


def test_captura_bate_com_o_modelo(tiny_qwen):
    ids = torch.randint(96, (1, 20), generator=torch.Generator().manual_seed(1))
    k = 2
    states = capture(tiny_qwen, ids, k, key_mode="raw")
    out = tiny_qwen(ids, output_hidden_states=True)

    torch.testing.assert_close(states.h_in, out.hidden_states[k - 1][0])
    torch.testing.assert_close(states.key, rms_normalize(out.hidden_states[k - 1][0]))
    torch.testing.assert_close(states.shallow, out.hidden_states[k][0])
    # O alvo é o estado final antes do Norm: relido pela cabeça, dá os logits finais.
    # (Tolerância folgada: o eps do Norm pesa nos estados pequenos de um modelo aleatório.)
    torch.testing.assert_close(hf_readout(tiny_qwen)(states.target), out.logits[0], atol=1e-3, rtol=1e-3)
    assert torch.equal(states.final_top, out.logits[0].argmax(-1))
    torch.testing.assert_close(states.final_nll[:-1], F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none"))
    assert states.final_nll[-1].isnan() and states.next_ids[-1] == -1
    assert (states.k, states.n_layers) == (2, 4)


def test_captura_com_tradutor(tiny_qwen):
    ids = torch.randint(96, (1, 10))
    lens = TunedLens(n_layers=4, d_model=32)
    with torch.no_grad():
        lens.translators[0].bias.fill_(0.5)
    states = capture(tiny_qwen, ids, k=1, lens=lens)
    plain = capture(tiny_qwen, ids, k=1)
    torch.testing.assert_close(states.shallow, plain.shallow + 0.5)


def test_k_fora_do_intervalo(tiny_qwen):
    with pytest.raises(ValueError):
        capture(tiny_qwen, torch.zeros(1, 4, dtype=torch.long), k=4)


def synthetic_text(n_patterns: int = 8, repeats: int = 10, noise: float = 0.05, seed: int = 0):
    """Texto sintético: `n_patterns` situações que se repetem `repeats` vezes, embaralhadas.

    A camada rasa (aleatória) fica difusa, com estresse alto; o estado final
    aponta para a linha da cabeça da resposta e é nítido, com estresse baixo.
    """
    g = torch.Generator().manual_seed(seed)
    head = torch.randn(VOCAB, D, generator=g) / D**0.5
    answers = torch.randperm(VOCAB, generator=g)[:n_patterns]
    keys = torch.randn(n_patterns, D, generator=g)
    shallow = torch.randn(n_patterns, D, generator=g)
    order = torch.cat([torch.randperm(n_patterns, generator=g) for _ in range(repeats)])

    def jitter(x):
        return x[order] + noise * torch.randn(len(order), D, generator=g)

    def readout(q):
        return rms_normalize(q) @ head.T

    target = rms_normalize(head[answers][order])
    final_logits = readout(target)
    next_ids = final_logits.argmax(-1)
    h_in = jitter(keys)
    states = LayerStates(
        h_in=h_in,
        key=rms_normalize(h_in),
        shallow=jitter(shallow),
        target=target,
        final_top=final_logits.argmax(-1),
        final_nll=F.cross_entropy(final_logits, next_ids, reduction="none"),
        next_ids=next_ids,
        k=17,
        n_layers=24,
    )
    return states, readout


def test_um_passo_com_eta_1_copia_a_distribuicao_final():
    states, readout = synthetic_text(n_patterns=1, repeats=2, noise=0.0)
    synapse = FastSynapse(D, D, rank=16, eta=1.0, gamma=1.0)
    simulate(states, readout, synapse, StressGate(tau=0.0), learn=True)
    q = states.shallow[1:2] + synapse(states.key[1:2])
    torch.testing.assert_close(readout(q), readout(states.target[1:2]), atol=1e-3, rtol=1e-3)


def test_curva_de_mielinizacao_sintetica():
    states, readout = synthetic_text()
    gate = StressGate(tau=0.3)
    # Posto 32 para 8 padrões: com posto 16, as chaves interferem, parte dos tokens
    # para logo acima de τ e, como saída cedo não ensina, ali fica.
    myelin = simulate(states, readout, FastSynapse(D, D, rank=32, eta=0.5, gamma=1.0), gate)
    static = simulate(states, readout, FastSynapse(D, D, rank=16), gate, learn=False)

    first, *_, last = myelin.blocks(5)
    assert first["exit_rate"] < 0.2
    assert last["exit_rate"] > 0.8 and last["exit_precision"] > 0.95
    assert first["depth"] > 23 and last["depth"] < 18  # de quase 24 camadas para k = 17
    assert all(row["exit_rate"] == 0.0 for row in static.blocks(5))


def test_saida_cedo_nao_ensina_a_sinapse():
    states, readout = synthetic_text()
    synapse = FastSynapse(D, D)
    log = simulate(states, readout, synapse, StressGate(tau=1.0))
    assert log.exit.all() and not log.audit.any()
    assert torch.equal(synapse.B, torch.zeros_like(synapse.B))
    assert torch.equal(log.depth, torch.full_like(log.depth, 17))


def test_auditoria_acorda_o_cortex_e_ensina():
    states, readout = synthetic_text()
    synapse = FastSynapse(D, D)
    log = simulate(states, readout, synapse, StressGate(tau=1.0), audit_rate=1.0)
    assert not log.exit.any() and log.audit.all() and log.agree.all()
    assert synapse.B.abs().sum() > 0


def test_blocos_resumem_o_texto():
    states, readout = synthetic_text(n_patterns=8, repeats=10)
    rows = simulate(states, readout, FastSynapse(D, D), StressGate()).blocks(4)
    assert [row["tokens"] for row in rows] == [20, 20, 20, 20]
    assert all(17 <= row["depth"] <= 24 for row in rows)
