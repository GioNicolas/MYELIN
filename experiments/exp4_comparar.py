"""Experimento 4b: tradutor estreito (en-py) contra tradutor amplo (multi), na mesma taxa de saída.

Para cada grupo de textos, quatro perguntas (ΔNLL em nats; negativo = a 1ª linha é melhor):
  1. estático amplo − estático estreito: quanto o tradutor amplo resolve sozinho, offline;
  2. MYELIN estreito − estático amplo: adaptar durante a leitura contra treinar melhor antes;
  3. MYELIN amplo − estático amplo: o ganho que sobra com o tradutor amplo;
  4. MYELIN amplo − estático estreito: os dois juntos.

    python experiments/exp4_comparar.py results/exp4_Qwen2.5-0.5B_k17.json results/exp4_Qwen2.5-0.5B_k17_multi.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp3_analise import gaps_vs_reference

MYELIN = ["viés lento + aud 5%", "kNN 256 + aud 5%", "viés lento + kNN 256 + aud 5%", "cache neural 256"]


def compare(narrow: dict, broad: dict) -> None:
    taus = [str(t) for t in narrow["config"]["taus"]]
    rows = defaultdict(lambda: defaultdict(list))
    for name, text in narrow["texts"].items():
        n, b = text["runs"], broad["texts"][name]["runs"]
        group = text["group"]
        pairs = {"estático amplo − estático estreito": (b["estático"], n["estático"])}
        for config in MYELIN:
            pairs[f"{config}: estreito − estático amplo"] = (n[config], b["estático"])
            pairs[f"{config}: amplo − estático amplo"] = (b[config], b["estático"])
            pairs[f"{config}: amplo − estático estreito"] = (b[config], n["estático"])
        for label, (run, reference) in pairs.items():
            gaps = gaps_vs_reference(run, reference, taus)
            rows[label][group].append(float(np.mean(gaps)) if gaps else float("nan"))

    groups = list(dict.fromkeys(t["group"] for t in narrow["texts"].values()))
    print(f"\n{'comparação (mesma taxa de saída)':<62} " + " ".join(f"{g:>13}" for g in groups))
    for label, by_group in rows.items():
        print(f"{label:<62} " + " ".join(f"{np.nanmean(by_group[g]):>+13.3f}" for g in groups))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    compare(*(json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:3]))
