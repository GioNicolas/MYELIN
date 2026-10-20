"""Experimento 1: a Curva de Mielinização no Qwen real (modo simulação).

Compara, no mesmo texto, na mesma camada k e com o mesmo tradutor P_k:

  - estático : Early-Exit Estático (sinapse zerada, nunca aprende), a linha de
               base que já inclui o efeito do contexto (atenção copiando padrões);
  - myelin   : Sinapse Rápida viva, consolidando o resíduo do córtex sempre que
               ele acorda; variando o modo das chaves, η, o posto e a
               auditoria (acordar o córtex por sorteio).

O texto é dividido em blocos consecutivos. Por bloco: taxa de saída cedo,
precisão das saídas (concordam com a final?), concordância da camada rasa
(saindo ou não) e perplexidade das previsões usadas. A pergunta do H1/H2 é
se a curva do MYELIN sobe além da do estático.

    python experiments/exp1_curva_mielina.py --lens results/tuned_lens_Qwen2.5-0.5B.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import FastSynapse, StressGate, TunedLens, capture, simulate  # noqa: E402
from myelin.wrapper import KEY_MODES, hf_readout, key_similarity, make_keys  # noqa: E402

DEFAULT_TEXTS = [
    ROOT / "src" / "myelin" / "fast_synapse.py",
    ROOT / "tests" / "test_fast_synapse.py",
    ROOT / "README.md",
]


def run_configs(
    key_modes: list[str], etas: list[float], ranks: list[int], audit_rates: list[float]
) -> list[dict]:
    """Estático primeiro; depois o MYELIN em cada combinação de chave, η, posto e auditoria."""
    base = {"key_mode": key_modes[0], "eta": etas[0], "rank": ranks[0]}
    configs = [{"name": "estático", "learn": False, "audit_rate": 0.0, **base}]
    for key_mode, eta, rank, audit in product(key_modes, etas, ranks, audit_rates):
        name = f"{key_mode[:5]} η={eta:g} r={rank}" + (f" aud={audit:.0%}" if audit else "")
        configs.append(
            {"name": name, "learn": True, "key_mode": key_mode, "eta": eta, "rank": rank, "audit_rate": audit}
        )
    return configs


def _pct(x: float) -> str:
    return "  —" if math.isnan(x) else f"{x:>3.0%}"


def print_text_table(title: str, full_ppl: float, runs: dict[str, dict]) -> None:
    width = max(34, 4 * len(next(iter(runs.values()))["blocks"]))
    print(f"\n{title} | modelo completo: ppl {full_ppl:.2f}")
    print(
        f"{'config':<28} | {'saída cedo por bloco (1º → último)':<{width}} | "
        f"{'precisão':>8} | {'rasa concorda 1º → últ.':>23} | {'camadas':>7} | {'ppl':>6}"
    )
    for name, run in runs.items():
        blocks, total = run["blocks"], run["summary"]
        curve = " ".join(_pct(b["exit_rate"]) for b in blocks)
        print(
            f"{name:<28} | {curve:<{width}} |{_pct(total['exit_precision']):>8} | "
            f"{_pct(blocks[0]['shallow_agree']):>11} → {_pct(blocks[-1]['shallow_agree']):<9} | "
            f"{total['depth']:>7.1f} | {total['ppl']:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k treinado (exp0b_tuned_lens.py)")
    parser.add_argument("--k", type=int, default=17, help="camada rasa (Experimento 0b: 17 de 24)")
    parser.add_argument("--texts", nargs="*", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--metric", default="entropy", choices=["entropy", "margin"])
    parser.add_argument("--key-modes", nargs="+", default=["center", "standardize"], choices=KEY_MODES)
    parser.add_argument("--etas", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--ranks", nargs="+", type=int, default=[64])
    parser.add_argument("--audit-rates", nargs="+", type=float, default=[0.0, 0.05])
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    readout, gate = hf_readout(model), StressGate(args.tau, args.metric)
    configs = run_configs(args.key_modes, args.etas, args.ranks, args.audit_rates)
    print(
        f"Modelo {args.model}: {n_layers} camadas, k={args.k}, τ={args.tau} ({args.metric}), "
        f"γ={args.gamma}, device={args.device}"
    )

    results = {"config": vars(args), "texts": {}}
    for path in args.texts:
        path = path.resolve()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
        ids = tokenizer(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids
        ids = ids[:, : args.max_tokens].to(args.device)
        states = capture(model, ids, args.k, lens, key_mode="raw")
        full_ppl = states.final_nll[:-1].mean().exp().item()
        keys = {mode: make_keys(states.h_in, mode) for mode in KEY_MODES}
        similarity = {mode: key_similarity(key) for mode, key in keys.items()}

        runs = {}
        for cfg in configs:
            start = time.time()
            synapse = FastSynapse(d_model, d_model, cfg["rank"], cfg["eta"], args.gamma).to(args.device)
            log = simulate(
                replace(states, key=keys[cfg["key_mode"]]),
                readout,
                synapse,
                gate,
                learn=cfg["learn"],
                audit_rate=cfg["audit_rate"],
            )
            runs[cfg["name"]] = {
                **cfg,
                "blocks": log.blocks(args.blocks),
                "summary": log.blocks(1)[0],
                "seconds": time.time() - start,
            }
        print_text_table(f"{name} ({ids.shape[1]} tokens)", full_ppl, runs)
        print("|cos| médio entre chaves: " + ", ".join(f"{m} {v:.2f}" for m, v in similarity.items()))
        results["texts"][name] = {"full_ppl": full_ppl, "key_similarity": similarity, "runs": runs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"exp1_{args.model.split('/')[-1]}_k{args.k}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
