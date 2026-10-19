import torch

from myelin import TunedLens


def test_comeca_como_logit_lens():
    lens = TunedLens(n_layers=4, d_model=8)
    h = torch.randn(5, 8)
    for layer in range(1, 4):
        assert torch.equal(lens(layer, h), h)


def test_salva_e_carrega(tmp_path):
    lens = TunedLens(n_layers=3, d_model=6)
    with torch.no_grad():
        lens.translators[1].weight.normal_()
    lens.save(tmp_path / "lens.pt", model="toy")
    loaded = TunedLens.load(tmp_path / "lens.pt")
    h = torch.randn(2, 6)
    assert torch.equal(loaded(2, h), lens(2, h))
