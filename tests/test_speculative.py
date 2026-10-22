"""Testes do rascunho especulativo, com os estados sintéticos de test_wrapper."""

from __future__ import annotations

import pytest
import torch

from myelin import KNNCache, NeuralCache
from myelin.speculative import draft, expected_speedup
from test_wrapper import synthetic_text


def test_rascunho_igual_ao_final_e_aceito_sempre():
    states, readout = synthetic_text()
    log = draft(states.__class__(**{**states.__dict__, "shallow": states.target}), readout)
    assert log.greedy.all()
    torch.testing.assert_close(log.alpha, torch.ones_like(log.alpha), atol=1e-4, rtol=0)


def test_aceitacao_e_uma_probabilidade_e_bate_com_o_top1():
    states, readout = synthetic_text()
    log = draft(states, readout)
    assert ((log.alpha >= 0) & (log.alpha <= 1 + 1e-5)).all()
    assert torch.equal(log.greedy, readout(states.shallow).argmax(-1) == states.final_top)


def test_memoria_aprende_com_o_verificador_e_sobe_a_aceitacao():
    states, readout = synthetic_text(n_patterns=8, repeats=10)
    static = draft(states, readout)
    learned = draft(states, readout, memory=KNNCache())
    last = slice(len(states.key) // 2, None)
    assert learned.greedy[last].float().mean() > 0.95 > static.greedy[last].float().mean()
    assert learned.alpha[last].mean() > static.alpha[last].mean() + 0.3


def test_cache_neural_no_rascunho():
    states, readout = synthetic_text(n_patterns=8, repeats=10)
    log = draft(states, readout, cache=NeuralCache(lam=0.5))
    assert log.greedy[len(states.key) // 2 :].all()


def test_speedup_esperado():
    # Sem aceitação, o melhor é γ = 1 e fica mais lento que o modelo completo.
    speedup, gamma = expected_speedup(0.0, k=12, n_layers=24, head=0.0)
    assert gamma == 1 and speedup == pytest.approx(1.0)
    # Aceitação total, cabeça grátis, k = L/2: γ = 8 dá 9 tokens por 8·0,5 + 0,5 = 4,5 passos.
    speedup, gamma = expected_speedup(1.0, k=12, n_layers=24, head=0.0)
    assert gamma == 8 and speedup == pytest.approx(2.0)
    # A cabeça cara piora o speedup.
    assert expected_speedup(0.8, 12, 24, 9.0)[0] < expected_speedup(0.8, 12, 24, 0.0)[0]
