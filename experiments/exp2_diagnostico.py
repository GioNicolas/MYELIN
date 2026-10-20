"""Experimento 2: a memória rápida aprende algo útil, antes de envolver o portão?

Sugerido pelas revisões externas da rodada 1 do Experimento 1. O córtex acorda
num calendário FIXO (a cada `--teacher-every` tokens), igual para todas as
memórias, e mede-se a camada rasa em todos os tokens, sempre ANTES de a memória
aprender com aquele token. Assim se separa "a memória generaliza para os
tokens seguintes?" de "o portão escolhe bem quando acordar?" (o colapso).

Memórias comparadas, todas lendo em cima do mesmo tradutor P_k:
  - estático     : nenhuma (B = 0), a linha de base;
  - viés online  : a mesma chave para todo token (sem contexto). Se ela
                   empatar com as outras, o ganho é só correção global do P_k
                   (ex.: o idioma), e não aprendizado do contexto;
  - NLMS r=64    : a Sinapse Rápida da rodada 1, com chaves centradas (como
                   antes), padronizadas por estatísticas fixas, da entrada
                   h^(k-1) ou da saída h^(k) da camada k;
  - NLMS r=d     : posto completo (sem jogar fora parte da chave);
  - kNN          : memória exata dos pares (chave, correção), o teto não
                   paramétrico (cresce com o texto, não é o MYELIN);
  - kNN 256      : o mesmo, guardando só as 256 últimas consultas (tamanho fixo);
  - viés + kNN   : correção global e, por cima, a específica de cada situação.

Resultado principal: diferença de NLL da camada rasa (próximo token real) na
2ª metade de cada texto, em relação ao estático. Negativo = a memória ajudou.

    python experiments/exp2_diagnostico.py --lens results/tuned_lens_Qwen2.5-0.5B.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import FastSynapse, KNNCache, OnlineBias, StackedMemory, StressGate, TunedLens, capture, simulate  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout, key_interference, make_keys  # noqa: E402
from textos import calibration_ids, eval_texts  # noqa: E402

CONFIGS = [
    {"name": "estático", "memory": None, "key": "bias"},
    {"name": "viés online", "memory": "bias", "key": "bias"},
    {"name": "r=64 entrada centrada", "memory": "nlms", "key": "input-center", "rank": 64},
    {"name": "r=64 entrada fixa", "memory": "nlms", "key": "input-fixed", "rank": 64},
    {"name": "r=64 saída fixa", "memory": "nlms", "key": "output-fixed", "rank": 64},
    {"name": "r=d entrada fixa", "memory": "nlms", "key": "input-fixed", "rank": "d"},
    {"name": "r=d saída fixa", "memory": "nlms", "key": "output-fixed", "rank": "d"},
    {"name": "kNN entrada fixa", "memory": "knn", "key": "input-fixed"},
    {"name": "kNN saída fixa", "memory": "knn", "key": "output-fixed"},
    {"name": "kNN 256 saída fixa", "memory": "knn256", "key": "output-fixed"},
    {"name": "viés + kNN 256 saída", "memory": "bias+knn256", "key": "output-fixed"},
]


def build_memory(cfg: dict, d_model: int, eta: float):
    if cfg["memory"] == "knn":
        return KNNCache()
    if cfg["memory"] == "knn256":
        return KNNCache(capacity=256)
    if cfg["memory"] == "bias":
        return OnlineBias(d_model, eta=eta)
    if cfg["memory"] == "bias+knn256":
        return StackedMemory(OnlineBias(d_model, eta=eta), KNNCache(capacity=256))
    rank = d_model if cfg.get("rank") == "d" else cfg.get("rank", 16)
    return FastSynapse(d_model, d_model, rank=rank, eta=eta, gamma=1.0)


def second_half_nll(nll: torch.Tensor) -> float:
    half = nll[len(nll) // 2 :]
    return half[~half.isnan()].mean().item()


def _pct(x: float) -> str:
    return "  —" if math.isnan(x) else f"{x:>3.0%}"


def print_text_table(title: str, full_nll: float, runs: dict[str, dict]) -> None:
    static_nll = runs["estático"]["nll_2nd"]
    print(f"\n{title} | modelo completo, 2ª metade: NLL {full_nll:.2f} (ppl {math.exp(full_nll):.1f})")
    print(f"{'memória':<24} | {'camada rasa concorda c/ final, por bloco':<40} | {'ppl rasa 2ª met.':>16} | {'ΔNLL':>6}")
    for name, run in runs.items():
        curve = " ".join(_pct(b["shallow_agree"]) for b in run["blocks"])
        delta = run["nll_2nd"] - static_nll
        print(f"{name:<24} | {curve:<40} | {math.exp(run['nll_2nd']):>16.1f} | {delta:>+6.2f}")


def print_summary(results: dict) -> None:
    groups = sorted({t["group"] for t in results["texts"].values()})
    print("\nResumo: ΔNLL médio da camada rasa na 2ª metade vs. estático (negativo = ajudou)")
    print(f"{'memória':<24} | " + " | ".join(f"{g:>10}" for g in groups) + f" | {'todos':>6} | melhorou em")
    names = [name for name in next(iter(results["texts"].values()))["runs"] if name != "estático"]
    for name in names:
        deltas = {g: [] for g in groups}
        for text in results["texts"].values():
            runs = text["runs"]
            deltas[text["group"]].append(runs[name]["nll_2nd"] - runs["estático"]["nll_2nd"])
        everything = [d for ds in deltas.values() for d in ds]
        cells = " | ".join(f"{sum(ds) / len(ds):>+10.2f}" for ds in deltas.values())
        wins = sum(d < 0 for d in everything)
        print(f"{name:<24} | {cells} | {sum(everything) / len(everything):>+6.2f} | {wins}/{len(everything)} textos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k treinado (exp0b_tuned_lens.py)")
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--n-wikitext", type=int, default=2)
    parser.add_argument("--teacher-every", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--configs", nargs="*", help="nomes das memórias a rodar (padrão: todas)")
    parser.add_argument("--tag", default="", help="sufixo do arquivo de resultados (não sobrescrever rodadas)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    d_model = model.config.hidden_size
    readout, gate = hf_readout(model), StressGate()
    configs = [c for c in CONFIGS if not args.configs or c["name"] in args.configs or c["memory"] is None]
    print(
        f"Modelo {args.model}: k={args.k}, córtex a cada {args.teacher_every} tokens, η={args.eta}, γ=1, "
        f"device={args.device}"
    )

    stats = fit_key_stats(model, [ids.to(args.device) for ids in calibration_ids(tokenizer)], args.k)
    A64 = FastSynapse(d_model, d_model, rank=64).A.to(args.device)
    results = {"config": vars(args), "texts": {}}

    for text in eval_texts(tokenizer, args.max_tokens, args.n_wikitext):
        ids = text["ids"].to(args.device)
        states = capture(model, ids, args.k, lens)
        keys = {
            "bias": make_keys(states.h_in, "bias"),
            "input-center": make_keys(states.h_in, "center"),
            "input-fixed": make_keys(states.h_in, "fixed", stats["input"]),
            "output-fixed": make_keys(states.h_out, "fixed", stats["output"]),
        }
        interference = {
            key: {"r=d": key_interference(keys[key]), "r=64": key_interference(keys[key], A64)}
            for key in ("input-center", "input-fixed", "output-fixed")
        }

        runs = {}
        for cfg in configs:
            start = time.time()
            memory = build_memory(cfg, d_model, args.eta).to(args.device)
            log = simulate(
                replace(states, key=keys[cfg["key"]]),
                readout,
                memory,
                gate,
                learn=cfg["memory"] is not None,
                teacher_every=args.teacher_every,
            )
            runs[cfg["name"]] = {
                **cfg,
                "blocks": log.blocks(args.blocks),
                "nll_2nd": second_half_nll(log.shallow_nll),
                "seconds": time.time() - start,
            }

        full_nll = second_half_nll(states.final_nll)
        print_text_table(f"[{text['group']}] {text['name']} ({ids.shape[1]} tokens)", full_nll, runs)
        print(
            "|c| entre chaves (mediana / 99%), sem projeção e com r=64: "
            + "; ".join(
                f"{key} {v['r=d']['p50']:.2f}/{v['r=d']['p99']:.2f} e {v['r=64']['p50']:.2f}/{v['r=64']['p99']:.2f}"
                for key, v in interference.items()
            )
        )
        results["texts"][text["name"]] = {
            "group": text["group"],
            "full_nll_2nd": full_nll,
            "interference": interference,
            "runs": runs,
        }

    print_summary(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp2_{args.model.split('/')[-1]}_k{args.k}_every{args.teacher_every}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
