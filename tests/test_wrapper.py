"""Testes do acoplador em modo simulação.

Um Qwen2 minúsculo e aleatório testa a captura dos estados (roda em CPU em
menos de 1 s); estados sintéticos com padrões recorrentes testam a lógica do
portão e da Mielinização sem depender de um modelo treinado.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from myelin import FastSynapse, KNNCache, OnlineBias, StackedMemory, StressGate, TunedLens
from myelin.wrapper import (
    KeyStats,
    LayerStates,
    capture,
    causal_center,
    causal_standardize,
    fit_key_stats,
    hf_readout,
    key_interference,
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
    # Com 2 amostras, padronizar só deixa o sinal da diferença: por isso o modo "fixed".
    h2 = torch.tensor([[0.0, 5.0, -300.0, 2.0], [1.0, -5.0, 900.0, 2.5]])
    torch.testing.assert_close(causal_standardize(h2)[1], torch.tensor([1.0, -1.0, 1.0, 1.0]), atol=1e-3, rtol=0)


def test_padronizar_tira_o_dominio_de_uma_dimensao_gigante():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(200, 32, generator=g)
    h[:, 0] += 1000.0 * (torch.rand(200, generator=g) < 0.3)  # ativação gigante em 30% dos tokens
    # Centrar piora: todo token passa a desviar muito da média nessa dimensão,
    # e todas as chaves ficam quase paralelas. Padronizar resolve.
    assert key_interference(make_keys(h, "center"))["p50"] > 0.9
    assert key_interference(make_keys(h, "standardize"))["p99"] < 0.6 < key_interference(make_keys(h, "raw"))["p99"]
    stats = KeyStats(mean=h.mean(0), std=h.std(0))
    assert key_interference(make_keys(h, "fixed", stats))["p99"] < 0.6


def test_interferencia_depois_da_projecao():
    # Chaves ortogonais em d = 64 deixam de ser ortogonais num subespaço aleatório de posto 8.
    keys = torch.eye(64) * 8.0
    A = FastSynapse(64, 64, rank=8).A
    assert key_interference(keys, skip=0)["p99"] < 1e-6
    assert key_interference(keys, A, skip=0)["p50"] > 0.05


def test_chave_viés_e_a_mesma_para_todo_token():
    keys = make_keys(torch.randn(5, 4), "bias")
    assert torch.equal(keys, torch.ones(5, 4))


def test_modo_de_chave_desconhecido():
    with pytest.raises(ValueError):
        make_keys(torch.randn(3, 4), "whiten")
    with pytest.raises(ValueError):
        make_keys(torch.randn(3, 4), "fixed")


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


def test_estatisticas_de_calibracao(tiny_qwen):
    texts = [torch.randint(96, (1, 12)), torch.randint(96, (1, 9))]
    stats = fit_key_stats(tiny_qwen, texts, k=2, skip=4)
    hidden = [tiny_qwen(ids, output_hidden_states=True).hidden_states for ids in texts]
    h_in = torch.cat([h[1][0, 4:] for h in hidden])
    torch.testing.assert_close(stats["input"].mean, h_in.mean(0))
    torch.testing.assert_close(stats["input"].std, h_in.std(0).clamp_min(1e-3))
    assert stats["output"].mean.shape == (32,)


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
    h_in, shallow = jitter(keys), jitter(shallow)
    states = LayerStates(
        h_in=h_in,
        h_out=shallow,
        key=rms_normalize(h_in),
        shallow=shallow,
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


def test_memoria_exata_tambem_aprende_o_atalho():
    states, readout = synthetic_text()
    log = simulate(states, readout, KNNCache(beta=20.0, min_sim=0.5), StressGate(tau=0.3))
    # Aprende em um passo: no 1º bloco (2 repetições de cada padrão) a 2ª vez já sai cedo.
    first, *_, last = log.blocks(5)
    assert first["exit_rate"] <= 0.5 and last["exit_rate"] > 0.9
    assert log.blocks(1)[0]["exit_precision"] > 0.95


def test_memoria_exata_de_tamanho_fixo_guarda_as_ultimas():
    cache = KNNCache(capacity=3)
    keys = torch.eye(5, 8)
    cache.consolidate(keys, torch.arange(5.0).unsqueeze(-1).expand(5, 8))
    assert len(cache.keys) == 3
    torch.testing.assert_close(cache(keys[4:]), torch.full((1, 8), 4.0), atol=1e-4, rtol=0)
    assert torch.equal(cache(keys[:1]), torch.zeros(1, 8))  # esquecida: não reconhece mais


def test_vies_mais_knn_corrige_o_residuo_inteiro():
    memory = StackedMemory(OnlineBias(8, eta=0.5), KNNCache())
    key, residual = torch.randn(1, 8), torch.randn(1, 8)
    memory.consolidate(key, residual)
    torch.testing.assert_close(memory.memories[0].b, 0.5 * residual[0])
    torch.testing.assert_close(memory(key), residual)  # o kNN guarda a metade que o viés não cobriu
    other = torch.randn(1, 8)
    other = other - (other @ key.T) / (key @ key.T) * key  # chave ortogonal: só o viés vale
    torch.testing.assert_close(memory(other), 0.5 * residual)


def test_calendario_fixo_ignora_o_portao():
    states, readout = synthetic_text()
    synapse = FastSynapse(D, D, rank=32, eta=0.5, gamma=1.0)
    log = simulate(states, readout, synapse, StressGate(tau=1.0), teacher_every=4)
    assert torch.equal(~log.exit, torch.arange(len(log.exit)) % 4 == 0)
    assert not log.audit.any()
    # Só 1 em cada 4 tokens ensina, mas a memória generaliza para as repetições.
    first, *_, last = log.blocks(5)
    assert last["shallow_agree"] > first["shallow_agree"] + 0.3


def test_leitura_em_lote_igual_a_token_a_token():
    # Com γ = 1, decair por token não muda nada, mas força o caminho token a token.
    states, readout = synthetic_text()
    gate = StressGate(tau=0.3)
    logs = []
    for per_token in (False, True):
        memory = FastSynapse(D, D, rank=32, eta=0.5, gamma=1.0)
        logs.append(simulate(states, readout, memory, gate, teacher_every=4, decay_per_token=per_token))
    batched, step = logs
    assert torch.equal(batched.shallow_agree, step.shallow_agree) and torch.equal(batched.exit, step.exit)
    torch.testing.assert_close(batched.shallow_nll, step.shallow_nll, equal_nan=True)


def test_decaimento_por_token_continua_sem_consolidacao():
    states, readout = synthetic_text()
    synapse = FastSynapse(D, D, gamma=0.9)
    synapse.B.fill_(1.0)
    simulate(states, readout, synapse, StressGate(tau=1.0), decay_per_token=True)  # todos saem cedo
    torch.testing.assert_close(synapse.B, torch.full_like(synapse.B, 0.9 ** len(states.key)))
    synapse.B.fill_(1.0)
    simulate(states, readout, synapse, StressGate(tau=1.0))  # decaimento só na consolidação
    assert torch.equal(synapse.B, torch.ones_like(synapse.B))


def test_regiao_de_confianca_zero_e_o_estatico():
    states, readout = synthetic_text()
    gate = StressGate(tau=0.3)
    capped = simulate(states, readout, FastSynapse(D, D, rank=32), gate, max_correction=0.0)
    static = simulate(states, readout, FastSynapse(D, D, rank=32), gate, learn=False)
    assert torch.equal(capped.shallow_agree, static.shallow_agree)
    with pytest.raises(ValueError):
        simulate(states, readout, FastSynapse(D, D), gate, target_scale="final")


def test_blocos_resumem_o_texto():
    states, readout = synthetic_text(n_patterns=8, repeats=10)
    rows = simulate(states, readout, FastSynapse(D, D), StressGate()).blocks(4)
    assert [row["tokens"] for row in rows] == [20, 20, 20, 20]
    assert all(17 <= row["depth"] <= 24 for row in rows)
