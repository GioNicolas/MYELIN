"""Experimento 6: a camada rasa como rascunho de decodificação especulativa.

Early exit num modelo congelado não paga a conta (Experimentos 3 e 5). Aqui a
camada rasa só PROPÕE o próximo token e o modelo completo verifica todos: a
saída fica idêntica à do modelo completo e a pergunta vira a taxa de aceitação
do rascunho, que a memória rápida pode subir aprendendo com o verificador (que
dá o professor em todo token).

Mede, por texto e por grupo, com o tradutor amplo (Experimento 4b):
  - aceitação gulosa (top-1 igual) e na amostragem (Σ min(p_k, p_L));
  - speedup esperado pela fórmula padrão (ver myelin/speculative.py), com a
    cabeça cobrada em todo rascunho e a verificação custando um passo inteiro;
    também com a cabeça na proporção de um modelo de 7B (o Experimento 5 mostrou
    que a concordância depende da profundidade relativa, não do tamanho).

    python experiments/exp6_rascunho.py --lens results/tuned_lens_Qwen2.5-0.5B_multi.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import KNNCache, NeuralCache, TunedLens, capture  # noqa: E402
from myelin.speculative import draft, expected_speedup  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout, make_keys  # noqa: E402
from exp3_portao import head_cost_in_layers  # noqa: E402
from textos import calibration_ids, domain_texts  # noqa: E402

# Cabeça do Qwen2.5-7B: 3584 × 152064 pesos / ~233M por camada ≈ 2,3 de 28 camadas.
HEAD_FRACTION_7B = 2.3 / 28

CONFIGS = [
    {"name": "estático", "knn": None, "cache": False},
    {"name": "kNN 256", "knn": 256, "cache": False},
    {"name": "kNN 1024", "knn": 1024, "cache": False},
    {"name": "cache neural 256", "knn": None, "cache": True},
    {"name": "kNN 256 + cache neural", "knn": 256, "cache": True},
]


def run_metrics(log, k: int, n_layers: int, head: float) -> dict[str, float]:
    half = torch.arange(len(log.greedy)) >= len(log.greedy) // 2
    valid = ~log.shallow_nll.isnan()
    greedy, alpha = log.greedy.float().mean().item(), log.alpha.mean().item()
    return {
        "greedy": greedy,
        "greedy_2nd": log.greedy[half].float().mean().item(),
        "alpha": alpha,
        "alpha_2nd": log.alpha[half].mean().item(),
        "shallow_nll": log.shallow_nll[valid].mean().item(),
        "speedup_greedy": expected_speedup(greedy, k, n_layers, head)[0],
        "speedup_alpha": expected_speedup(alpha, k, n_layers, head)[0],
        "speedup_greedy_7b_head": expected_speedup(greedy, k, n_layers, HEAD_FRACTION_7B * n_layers)[0],
    }


def print_summary(results: dict) -> None:
    texts = list(results["texts"].values())
    groups = list(dict.fromkeys(t["group"] for t in texts))
    n_layers, head = results["n_layers"], results["head_cost_layers"]
    for k in results["config"]["ks"]:
        key = str(k)
        names = list(texts[0]["runs"][key])
        c_draft = (k + head) / (n_layers + head)
        print(f"\n=== k = {k} de {n_layers}: rascunho custa {c_draft:.2f} e verificação 1,00 de um passo do modelo completo ===")

        def mean(name: str, metric: str, group: str | None = None) -> float:
            chosen = [t for t in texts if group is None or t["group"] == group]
            return sum(t["runs"][key][name][metric] for t in chosen) / len(chosen)

        print("Aceitação gulosa (top-1 do rascunho == top-1 do modelo), média dos textos do grupo")
        print(f"{'config':<26} " + " ".join(f"{g:>13}" for g in groups) + f" {'todos':>7} {'α amostr.':>9} {'speedup gul.':>12} {'speedup α':>10} {'gul. c/ cabeça 7B':>17}")
        for name in names:
            greedy_all, alpha_all = mean(name, "greedy"), mean(name, "alpha")
            print(
                f"{name:<26} " + " ".join(f"{mean(name, 'greedy', g):>13.1%}" for g in groups)
                + f" {greedy_all:>7.1%} {alpha_all:>9.1%}"
                + f" {expected_speedup(greedy_all, k, n_layers, head)[0]:>11.2f}×"
                + f" {expected_speedup(alpha_all, k, n_layers, head)[0]:>9.2f}×"
                + f" {expected_speedup(greedy_all, k, n_layers, HEAD_FRACTION_7B * n_layers)[0]:>16.2f}×"
            )
        print("Ganho de aceitação gulosa sobre o estático (pontos percentuais), 2ª metade do texto")
        for name in names[1:]:
            gains = [100 * (mean(name, "greedy_2nd", g) - mean("estático", "greedy_2nd", g)) for g in groups]
            print(f"{name:<26} " + " ".join(f"{gain:>+13.1f}" for gain in gains))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k (de preferência o amplo, corpus multi)")
    parser.add_argument("--ks", nargs="+", type=int, default=[8, 12, 17], help="camadas do rascunho")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cache-lambda", type=float, default=0.1)
    parser.add_argument("--configs", nargs="*", help="nomes das configurações (padrão: todas)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.device == "cpu":
        print("Aviso: sem GPU este experimento leva muitas horas (o rascunho é token a token). Prefira T4.")
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    n_layers = model.config.num_hidden_layers
    head = head_cost_in_layers(model)
    readout = hf_readout(model)
    configs = [c for c in CONFIGS if not args.configs or c["name"] in args.configs or c["name"] == "estático"]
    print(f"Modelo {args.model}: {n_layers} camadas, cabeça ≈ {head:.1f} camadas, k ∈ {args.ks}, device={args.device}")

    calibration = [ids.to(args.device) for ids in calibration_ids(tokenizer)]
    stats = {k: fit_key_stats(model, calibration, k)["output"] for k in args.ks}
    results = {"config": vars(args), "n_layers": n_layers, "head_cost_layers": head, "texts": {}}
    start = time.time()

    for text in domain_texts(tokenizer, args.max_tokens):
        ids = text["ids"].to(args.device)
        runs = {}
        for k in args.ks:
            states = capture(model, ids, k, lens)
            states = replace(states, key=make_keys(states.h_out, "fixed", stats[k]))
            runs[str(k)] = {}
            for cfg in configs:
                memory = KNNCache(capacity=cfg["knn"]) if cfg["knn"] else None
                cache = NeuralCache(lam=args.cache_lambda) if cfg["cache"] else None
                log = draft(states, readout, memory=memory, cache=cache)
                runs[str(k)][cfg["name"]] = run_metrics(log, k, n_layers, head)
        results["texts"][text["name"]] = {"group": text["group"], "runs": runs}
        print(f"[{text['group']}] {text['name']}: pronto ({time.time() - start:.0f}s acumulados)", flush=True)

    print_summary(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp6_{args.model.split('/')[-1]}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
