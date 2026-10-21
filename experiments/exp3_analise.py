"""Experimento 3, análise: ganho da memória contra o estático na MESMA taxa de saída, por texto e por grupo.

Para cada texto, a curva do estático (taxa de saída → ΔNLL, 5 valores de τ) é
interpolada linearmente na taxa de saída de cada ponto da memória; pontos fora
da faixa do estático ficam de fora (sem extrapolação).

    python experiments/exp3_analise.py results/exp3_Qwen2.5-0.5B_k17.json
"""

import json, sys
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
r = json.load(open(sys.argv[1], encoding="utf-8"))
taus = [str(t) for t in r["config"]["taus"]]
texts = r["texts"]
names = list(next(iter(texts.values()))["runs"])
mem = [n for n in names if n != "estático"]

def interp(x, xs, ys):
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9: return float("nan")
    return float(np.interp(x, xs, ys))

# 1) por texto: ganho na mesma taxa de saída (média dos τ dentro da faixa)
print("ΔNLL(memória) − ΔNLL(estático) na mesma taxa de saída, média dos τ dentro da faixa do estático")
print(f"{'texto':<38} {'grupo':<11} " + " ".join(f"{n[:22]:>23}" for n in mem))
per_group = defaultdict(lambda: defaultdict(list))
for tname, t in texts.items():
    st = t["runs"]["estático"]
    xs = [st[x]["exit_rate"] for x in taus]; ys = [st[x]["delta_nll"] for x in taus]
    row = []
    for n in mem:
        gaps = [m["delta_nll"] - interp(m["exit_rate"], xs, ys) for m in t["runs"][n].values()]
        gaps = [g for g in gaps if g == g]
        g = np.mean(gaps) if gaps else float("nan")
        row.append(f"{g:+.3f} ({len(gaps)})"); per_group[t["group"]][n].append(g)
    print(f"{tname[-38:]:<38} {t['group']:<11} " + " ".join(f"{c:>23}" for c in row))
print("\nPor grupo:")
for g, d in per_group.items():
    print(f"  {g:<11} " + " ".join(f"{np.nanmean(d[n]):>+23.3f}" for n in mem))

# 2) por grupo e τ: tabela de pontos
print("\nPontos por grupo (média dos textos do grupo): saída% / ΔNLL / custo real")
groups = sorted({t["group"] for t in texts.values()})
for g in groups:
    ts = [t for t in texts.values() if t["group"] == g]
    print(f"\n[{g}] ({len(ts)} textos) NLL completo = {np.mean([t['full_nll'] for t in ts]):.3f}")
    for n in names:
        cells = []
        for x in taus:
            e = np.mean([t["runs"][n][x]["exit_rate"] for t in ts]); d = np.mean([t["runs"][n][x]["delta_nll"] for t in ts])
            cells.append(f"{e:4.0%} {d:+.2f}")
        print(f"  {n:<30} " + " | ".join(cells))

# 3) mielinização: saída 2ª metade vs total
print("\nTaxa de saída: total → 2ª metade (média dos 7 textos), τ=0.3")
for n in names:
    e = np.mean([t["runs"][n]["0.3"]["exit_rate"] for t in texts.values()]); e2 = np.mean([t["runs"][n]["0.3"]["exit_rate_2nd"] for t in texts.values()])
    print(f"  {n:<30} {e:.0%} → {e2:.0%}")
