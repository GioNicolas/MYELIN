"""Testes da simulação do ciclo real de decodificação especulativa (Experimento 7)."""

from __future__ import annotations

import pytest
import torch

from myelin import KNNCache, NeuralCache
from myelin.controls import LastCorrection, MeanCorrection, ShuffledKNN
from myelin.speculative import ChainDrafter, FixedGamma, HeadDrafter, PromptLookupDrafter, ReflexDrafter, run_cycles
from test_wrapper import D, synthetic_text


class Oracle:
    """Propõe sempre a continuação certa: o teto do ciclo."""

    def __init__(self, final_top: list[int]):
        self.final_top = final_top

    def propose(self, c, gamma):
        return self.final_top[c : c + gamma], 0.0

    def observe(self, positions):
        pass


class Silent:
    def propose(self, c, gamma):
        return [], 0.0

    def observe(self, positions):
        pass


def test_sem_rascunho_e_a_decodificacao_normal():
    stats = run_cycles(Silent(), list(range(100)), start=0, end=50, gamma=4, full_cost=10.0)
    assert stats.tokens == 50 and stats.cost == pytest.approx(50.0) and stats.speedup == pytest.approx(1.0)


def test_oraculo_emite_gamma_mais_um_por_ciclo():
    stats = run_cycles(Oracle(list(range(100))), list(range(100)), start=0, end=60, gamma=3, full_cost=10.0, extra_token_cost=0.0)
    assert stats.tokens == 60 and stats.cycles == 15  # 4 tokens por ciclo
    assert stats.speedup == pytest.approx(4.0) and stats.acceptance == pytest.approx(1.0)


def test_custo_do_rascunho_pela_camada():
    states, readout = synthetic_text()
    final_top = states.final_top.tolist()
    stats = run_cycles(HeadDrafter(states, readout, head=9.0), final_top, 0, len(final_top) - 1, gamma=1, full_cost=33.0, extra_token_cost=0.0)
    # Cada ciclo custa (k + H)/(L + H) + 1 = 26/33 + 1, com k = 17.
    assert stats.cost == pytest.approx(stats.cycles * (26 / 33 + 1))


def test_memoria_no_rascunho_aprende_com_o_verificador():
    states, readout = synthetic_text(n_patterns=8, repeats=10)
    final_top, end = states.final_top.tolist(), len(states.key) - 1
    static = run_cycles(HeadDrafter(states, readout, head=0.0), final_top, 0, end, gamma=2, full_cost=24.0)
    learned = run_cycles(HeadDrafter(states, readout, head=0.0, memory=KNNCache()), final_top, 0, end, gamma=2, full_cost=24.0)
    cache = run_cycles(
        HeadDrafter(states, readout, head=0.0, cache=NeuralCache(lam=0.9, min_sim=0.5)), final_top, 0, end, gamma=2, full_cost=24.0
    )
    assert learned.acceptance > static.acceptance + 0.3
    assert cache.acceptance > static.acceptance + 0.3


def test_busca_de_ngramas_acha_a_repeticao():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 6
    stats = run_cycles(PromptLookupDrafter(tokens, n_max=3), tokens[1:] + [0], start=0, end=len(tokens) - 1, gamma=4, full_cost=1.0)
    assert stats.speedup > 2.5  # depois da 1ª repetição, quase tudo é proposto e aceito


def test_reflexo_reconhece_a_situacao_pelo_estado_oculto():
    torch.manual_seed(0)
    situations = torch.randn(8, D)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 6
    keys = situations[[t - 1 for t in tokens]] + 0.01 * torch.randn(len(tokens), D)
    drafter = ReflexDrafter(keys, tokens, tau=0.9)
    stats = run_cycles(drafter, tokens[1:] + [0], start=0, end=len(tokens) - 1, gamma=4, full_cost=1.0)
    assert stats.speedup > 2.5
    # Chaves sem nenhuma semelhança: o reflexo não especula e custa o mesmo que decodificar normalmente.
    lost = ReflexDrafter(torch.randn(len(tokens), D), tokens, tau=0.9)
    assert run_cycles(lost, tokens[1:] + [0], 0, len(tokens) - 1, gamma=4, full_cost=1.0).speedup == pytest.approx(1.0)


def test_corrente_usa_o_segundo_quando_o_primeiro_cala():
    tokens = list(range(40))
    stats = run_cycles(ChainDrafter(Silent(), Oracle(tokens[1:] + [0])), tokens[1:] + [0], 0, 39, gamma=1, full_cost=1.0, extra_token_cost=0.0)
    assert stats.speedup == pytest.approx(2.0, rel=0.05)


def test_controles_da_memoria():
    keys = torch.eye(3, D)
    last, mean = LastCorrection(), MeanCorrection()
    for memory in (last, mean):
        memory.consolidate(keys[:1], torch.full((1, D), 1.0))
        memory.consolidate(keys[1:2], torch.full((1, D), 3.0) - memory(keys[1:2]))
    assert torch.allclose(last(keys[2:]), torch.full((1, D), 3.0))
    assert torch.allclose(mean(keys[2:]), torch.full((1, D), 2.0))
    # O embaralhado guarda as correções verdadeiras, mas as devolve fora de ordem.
    shuffled = ShuffledKNN(beta=50.0, seed=0)
    for i, value in enumerate((1.0, 5.0, 9.0)):
        shuffled(keys[i : i + 1])
        shuffled.consolidate(keys[i : i + 1], torch.full((1, D), value) - shuffled._applied)
    assert sorted(float(c[0]) for c in shuffled.corrections) == [1.0, 5.0, 9.0]


def test_reflexo_com_dois_tokens_e_custo_da_busca():
    # O token 9 aparece depois de 1 e depois de 2; só o par (2, 9) prevê o 7.
    tokens = [1, 9, 5, 2, 9, 7, 3, 2, 9]
    keys = torch.ones(len(tokens), D)  # todas as situações parecem iguais: só os tokens decidem
    c = len(tokens) - 1
    assert ReflexDrafter(keys, tokens, tau=0.5, min_match=1).propose(c, 1)[0] in ([5], [7])
    drafts, cost = ReflexDrafter(keys, tokens, tau=0.5, min_match=2, candidate_cost=0.01).propose(c, 1)
    assert drafts == [7] and cost == pytest.approx((c - 1) * 0.01)


def test_cada_elo_da_corrente_usa_o_seu_gamma():
    tokens = list(range(40))
    oracle = Oracle(tokens[1:] + [0])
    stats = run_cycles(ChainDrafter(Silent(), FixedGamma(oracle, 1)), tokens[1:] + [0], 0, 39, gamma=6, full_cost=1.0, extra_token_cost=0.0)
    assert stats.speedup == pytest.approx(2.0, rel=0.05)  # 2 tokens por ciclo, não 7
