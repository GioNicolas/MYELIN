"""Experimento 4: a memória rápida como adaptação ao domínio, em mais textos e contra o cache neural.

O Experimento 3 mostrou que o ganho do MYELIN aparece onde o tradutor P_k erra
de forma sistemática (português, código do próprio projeto) e some em inglês.
Aqui isso vira a pergunta principal, com 15 textos mais longos (2048 tokens),
3 por grupo:
  - fora do treino do tradutor: português, outros idiomas (es, de, fr),
    código não-Python (Rust, Go, JavaScript);
  - controles, do mesmo tipo do treino: Python, prosa em inglês (WikiText).

Mesmo portão e mesmas curvas custo × qualidade do Experimento 3, com duas
linhas de base novas:
  - viés lento sozinho: se ele empatar com viés + kNN, o ganho é só correção
    global (um viés), e não memória do contexto;
  - cache neural (Grave et al., 2017): guarda o próximo token real de cada
    token e mistura na previsão rasa, sem precisar do córtex.

    python experiments/exp4_dominio.py --lens results/tuned_lens_Qwen2.5-0.5B.pt
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

from myelin import FastSynapse, KNNCache, NeuralCache, OnlineBias, StackedMemory, StressGate, TunedLens, capture, simulate  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout, make_keys  # noqa: E402
from exp3_analise import analyze  # noqa: E402
from exp3_portao import head_cost_in_layers, print_summary, run_metrics  # noqa: E402
from textos import calibration_ids, domain_texts  # noqa: E402

CONFIGS = [
    {"name": "estático", "memory": None, "cache": False, "audit": 0.0},
    {"name": "viés lento + aud 5%", "memory": "bias", "cache": False, "audit": 0.05},
    {"name": "kNN 256 + aud 5%", "memory": "knn256", "cache": False, "audit": 0.05},
    {"name": "viés lento + kNN 256 + aud 5%", "memory": "bias+knn256", "cache": False, "audit": 0.05},
    {"name": "cache neural 256", "memory": None, "cache": True, "audit": 0.0},
]


def build_memory(cfg: dict, d_model: int, args):
    if cfg["memory"] == "bias":
        return OnlineBias(d_model, eta=args.bias_eta)
    if cfg["memory"] == "knn256":
        return KNNCache(capacity=256)
    if cfg["memory"] == "bias+knn256":
        return StackedMemory(OnlineBias(d_model, eta=args.bias_eta), KNNCache(capacity=256))
    return FastSynapse(d_model, d_model, rank=16)  # sem memória: nunca aprende


def quarter_metrics(log, full_nll: torch.Tensor, n: int = 4) -> list[dict[str, float]]:
    """Taxa de saída e ΔNLL em cada quarto do texto: a adaptação acumula?"""
    rows = []
    for exit, nll, full in zip(log.exit.chunk(n), log.nll.chunk(n), full_nll.chunk(n)):
        valid = ~nll.isnan()
        rows.append({"exit_rate": exit.float().mean().item(), "delta_nll": (nll[valid] - full[valid]).mean().item()})
    return rows


def print_quarters(results: dict, tau: str) -> None:
    texts = list(results["texts"].values())
    names = list(texts[0]["runs"])
    print(f"\nPor quarto do texto, τ = {tau}: saída / ΔNLL (média dos textos do grupo)")
    for group in dict.fromkeys(t["group"] for t in texts):
        group_texts = [t for t in texts if t["group"] == group]
        print(f"\n[{group}]")
        for name in names:
            cells = []
            for q in range(4):
                exit_rate = sum(t["runs"][name][tau]["quarters"][q]["exit_rate"] for t in group_texts) / len(group_texts)
                delta = sum(t["runs"][name][tau]["quarters"][q]["delta_nll"] for t in group_texts) / len(group_texts)
                cells.append(f"{exit_rate:4.0%} {delta:+.2f}")
            print(f"  {name:<30} " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k treinado (exp0b_tuned_lens.py)")
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--bias-eta", type=float, default=0.03, help="η do viés online (lento: é uma média)")
    parser.add_argument("--cache-lambda", type=float, default=0.1, help="peso λ do cache neural na mistura")
    parser.add_argument("--cache-beta", type=float, default=20.0, help="nitidez β do cache neural")
    parser.add_argument("--configs", nargs="*", help="nomes das configurações (padrão: todas)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.device == "cpu":
        print("Aviso: sem GPU este experimento leva muitas horas (a leitura é token a token). Prefira T4 no Colab ou no Kaggle.")
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    head = head_cost_in_layers(model)
    readout = hf_readout(model)
    configs = [c for c in CONFIGS if not args.configs or c["name"] in args.configs or c["name"] == "estático"]
    print(
        f"Modelo {args.model}: {n_layers} camadas, k={args.k}, cabeça ≈ {head:.1f} camadas, "
        f"τ ∈ {args.taus}, {args.max_tokens} tokens por texto, device={args.device}"
    )

    stats = fit_key_stats(model, [ids.to(args.device) for ids in calibration_ids(tokenizer)], args.k)
    results = {"config": vars(args), "head_cost_layers": head, "texts": {}}
    start = time.time()

    for text in domain_texts(tokenizer, args.max_tokens):
        ids = text["ids"].to(args.device)
        states = capture(model, ids, args.k, lens)
        states = replace(states, key=make_keys(states.h_out, "fixed", stats["output"]))
        full_nll = states.final_nll.cpu()
        runs = {}
        for cfg in configs:
            runs[cfg["name"]] = {}
            for tau in args.taus:
                learn = cfg["memory"] is not None
                cache = NeuralCache(beta=args.cache_beta, lam=args.cache_lambda) if cfg["cache"] else None
                log = simulate(
                    states,
                    readout,
                    build_memory(cfg, d_model, args).to(args.device),
                    StressGate(tau),
                    learn=learn,
                    audit_rate=cfg["audit"],
                    decay_per_token=learn,
                    cache=cache,
                )
                metrics = run_metrics(log, args.k, n_layers, head, full_nll)
                metrics["quarters"] = quarter_metrics(log, full_nll)
                runs[cfg["name"]][str(tau)] = metrics
        valid = ~full_nll.isnan()
        results["texts"][text["name"]] = {"group": text["group"], "full_nll": full_nll[valid].mean().item(), "runs": runs}
        print(f"[{text['group']}] {text['name']}: pronto ({time.time() - start:.0f}s acumulados)", flush=True)

    print_summary(results, args.taus)
    analyze(results)
    print_quarters(results, str(args.taus[len(args.taus) // 2]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp4_{args.model.split('/')[-1]}_k{args.k}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Resultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
