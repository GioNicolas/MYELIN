"""Análise dos Experimentos 3 e 4: ganho de cada configuração contra o estático na MESMA taxa de saída.

Para cada texto, a curva do estático (taxa de saída → ΔNLL, um ponto por τ) é
interpolada linearmente na taxa de saída de cada ponto da configuração; pontos
fora da faixa do estático ficam de fora (sem extrapolação). Comparar pelo mesmo
τ engana: com memória, mais tokens saem cedo no mesmo τ.

    python experiments/exp3_analise.py results/exp3_Qwen2.5-0.5B_k17.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return float("nan")
    return float(np.interp(x, xs, ys))


def gains_at_equal_exit(text: dict, taus: list[str], name: str) -> list[float]:
    """ΔNLL(config) − ΔNLL(estático interpolado), um valor por τ dentro da faixa do estático."""
    static = text["runs"]["estático"]
    xs = [static[t]["exit_rate"] for t in taus]
    ys = [static[t]["delta_nll"] for t in taus]
    gaps = [m["delta_nll"] - _interp(m["exit_rate"], xs, ys) for m in text["runs"][name].values()]
    return [g for g in gaps if g == g]


def analyze(results: dict) -> None:
    taus = [str(t) for t in results["config"]["taus"]]
    texts = results["texts"]
    names = list(next(iter(texts.values()))["runs"])
    others = [n for n in names if n != "estático"]
    groups = list(dict.fromkeys(t["group"] for t in texts.values()))

    print("\nΔNLL(config) − ΔNLL(estático) na mesma taxa de saída (média dos τ dentro da faixa; negativo = melhor)")
    print(f"{'texto':<30} {'grupo':<13} " + " ".join(f"{n[:24]:>24}" for n in others))
    per_group = defaultdict(lambda: defaultdict(list))
    for text_name, text in texts.items():
        cells = []
        for name in others:
            gaps = gains_at_equal_exit(text, taus, name)
            gain = float(np.mean(gaps)) if gaps else float("nan")
            per_group[text["group"]][name].append(gain)
            cells.append(f"{gain:+.3f} ({len(gaps)})")
        print(f"{text_name[-30:]:<30} {text['group']:<13} " + " ".join(f"{c:>24}" for c in cells))
    print("\nPor grupo:")
    for group in groups:
        print(f"  {group:<41} " + " ".join(f"{np.nanmean(per_group[group][n]):>+24.3f}" for n in others))

    print("\nPontos por grupo (média dos textos do grupo): saída / ΔNLL, um por τ")
    for group in groups:
        group_texts = [t for t in texts.values() if t["group"] == group]
        full = np.mean([t["full_nll"] for t in group_texts])
        print(f"\n[{group}] ({len(group_texts)} textos) NLL do modelo completo = {full:.3f}")
        for name in names:
            cells = []
            for tau in taus:
                exit_rate = np.mean([t["runs"][name][tau]["exit_rate"] for t in group_texts])
                delta = np.mean([t["runs"][name][tau]["delta_nll"] for t in group_texts])
                cells.append(f"{exit_rate:4.0%} {delta:+.2f}")
            print(f"  {name:<30} " + " | ".join(cells))

    tau = taus[len(taus) // 2]
    print(f"\nTaxa de saída com τ = {tau}: total → 2ª metade (média dos {len(texts)} textos)")
    for name in names:
        total = np.mean([t["runs"][name][tau]["exit_rate"] for t in texts.values()])
        second = np.mean([t["runs"][name][tau]["exit_rate_2nd"] for t in texts.values()])
        print(f"  {name:<30} {total:.0%} → {second:.0%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    analyze(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
