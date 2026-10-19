import math

import torch

from myelin import StressGate, entropy_stress, margin_stress


def test_entropia_uniforme_e_estresse_maximo():
    assert torch.isclose(entropy_stress(torch.zeros(50)), torch.tensor(1.0))


def test_entropia_nitida_e_estresse_minimo():
    logits = torch.full((50,), -100.0)
    logits[3] = 100.0
    assert entropy_stress(logits) < 1e-6


def test_margem_empate_e_estresse_alto():
    logits = torch.tensor([5.0, 5.0, -math.inf, -math.inf])
    assert torch.isclose(margin_stress(logits), torch.tensor(1.0))


def test_portao_roteia_por_token():
    sharp = torch.tensor([20.0, 0.0, 0.0, 0.0])
    flat = torch.zeros(4)
    stress, wake = StressGate(tau=0.3)(torch.stack([sharp, flat]))
    assert stress.shape == (2,)
    assert wake.tolist() == [False, True]
