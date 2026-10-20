"""Experimento 3: o H2 com o portão de volta, qualidade × custo, MYELIN contra o Early-Exit Estático.

O portão de estresse decide, token a token, se a camada rasa k basta; quando o
córtex acorda (ou a auditoria o sorteia), a memória rápida aprende com ele.
Para cada limiar τ, a simulação inteira é refeita (a memória aprende coisas
diferentes com τ diferentes), e cada configuração vira uma curva:

    custo (camadas por token)  ×  qualidade (NLL no próximo token real)

O H2 se confirma se a curva do MYELIN ficar abaixo da do estático (menos NLL
com o mesmo custo). Duas contas de custo:
  - camadas: k na saída cedo, L com o córtex (a conta nominal);
  - custo real: inclui a cabeça d × |V|, que a leitura rasa paga em TODO token
    e o córtex paga de novo (no Qwen2.5-0.5B ela vale ~9 camadas).

    python experiments/exp3_portao.py --lens results/tuned_lens_Qwen2.5-0.5B.pt
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

from myelin import FastSynapse, KNNCache, OnlineBias, StackedMemory, StressGate, TunedLens, capture, simulate  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout, make_keys  # noqa: E402
from textos import calibration_ids, eval_texts  # noqa: E402

CONFIGS = [
    {"name": "estático", "memory": None, "key": "output-fixed", "audit": 0.0},
    {"name": "kNN 256", "memory": "knn256", "key": "output-fixed", "audit": 0.0},
    {"name": "kNN 256 + aud 5%", "memory": "knn256", "key": "output-fixed", "audit": 0.05},
    {"name": "viés lento + kNN 256 + aud 5%", "memory": "bias+knn256", "key": "output-fixed", "audit": 0.05},
    {"name": "r=64 centrada + aud 5%", "memory": "nlms64", "key": "input-center", "audit": 0.05},
]


def build_memory(cfg: dict, d_model: int, args):
    if cfg["memory"] == "knn256":
        return KNNCache(capacity=256)
    if cfg["memory"] == "bias+knn256":
        return StackedMemory(OnlineBias(d_model, eta=args.bias_eta), KNNCache(capacity=256))
    if cfg["memory"] == "nlms64":
        return FastSynapse(d_model, d_model, rank=64, eta=args.eta, gamma=args.gamma)
    return FastSynapse(d_model, d_model, rank=16)  # estático: nunca aprende


def head_cost_in_layers(model) -> float:
    """Pesos da cabeça d × |V| divididos pelos pesos de uma camada média."""
    n_layers = model.config.num_hidden_layers
    per_layer = sum(p.numel() for p in model.model.layers.parameters()) / n_layers
    return model.lm_head.weight.numel() / per_layer


def run_metrics(log, k: int, n_layers: int, head: float, full_nll: torch.Tensor) -> dict[str, float]:
    exit = log.exit.float()
    wake = 1.0 - exit
    cost = (k + head + wake * (n_layers - k + head)).mean().item() / (n_layers + head)
    valid = ~log.nll.isnan()
    half = torch.arange(len(exit)) >= len(exit) // 2
    exits = log.exit
    return {
        "exit_rate": exit.mean().item(),
        "exit_rate_2nd": exit[half].mean().item(),
        "depth": log.depth.mean().item(),
        "real_cost": cost,
        "delta_nll": (log.nll[valid] - full_nll[valid]).mean().item(),
        "delta_nll_2nd": (log.nll[valid & half] - full_nll[valid & half]).mean().item(),
        "exit_precision": log.agree[exits].float().mean().item() if exits.any() else float("nan"),
        "shallow_agree": log.shallow_agree.float().mean().item(),
        "audit_rate": log.audit.float().mean().item(),
    }


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:.0%}"


def print_summary(results: dict, taus: list[float]) -> None:
    texts = list(results["texts"].values())
    names = list(texts[0]["runs"])
    keys = ["exit_rate", "depth", "real_cost", "delta_nll", "delta_nll_2nd", "exit_precision"]
    print(f"\nResumo (média de {len(texts)} textos). ΔNLL = NLL das previsões usadas − NLL do modelo completo (menor = melhor)")
    print(
        f"{'config':<30} | {'τ':>4} | {'saída':>5} | {'camadas':>7} | {'custo real':>10} | "
        f"{'ΔNLL':>6} | {'ΔNLL 2ª':>7} | {'precisão':>8}"
    )
    for name in names:
        for tau in taus:
            m = {key: sum(t["runs"][name][str(tau)][key] for t in texts) / len(texts) for key in keys}
            print(
                f"{name:<30} | {tau:>4.2f} | {m['exit_rate']:>5.0%} | {m['depth']:>7.1f} | {m['real_cost']:>10.0%} | "
                f"{m['delta_nll']:>+6.3f} | {m['delta_nll_2nd']:>+7.3f} | {_pct(m['exit_precision']):>8}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k treinado (exp0b_tuned_lens.py)")
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--eta", type=float, default=0.5, help="η da sinapse linear")
    parser.add_argument("--gamma", type=float, default=0.999, help="γ da sinapse linear, aplicado a cada token")
    parser.add_argument("--bias-eta", type=float, default=0.03, help="η do viés online (lento: é uma média)")
    parser.add_argument("--configs", nargs="*", help="nomes das configurações (padrão: todas)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.device == "cpu":
        print("Aviso: sem GPU este experimento leva horas (a leitura é token a token). Prefira T4 no Colab ou no Kaggle.")
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    head = head_cost_in_layers(model)
    readout = hf_readout(model)
    configs = [c for c in CONFIGS if not args.configs or c["name"] in args.configs or c["memory"] is None]
    print(
        f"Modelo {args.model}: {n_layers} camadas, k={args.k}, cabeça ≈ {head:.1f} camadas, "
        f"τ ∈ {args.taus}, device={args.device}"
    )

    stats = fit_key_stats(model, [ids.to(args.device) for ids in calibration_ids(tokenizer)], args.k)
    results = {"config": vars(args), "head_cost_layers": head, "texts": {}}
    start = time.time()

    for text in eval_texts(tokenizer, args.max_tokens):
        ids = text["ids"].to(args.device)
        states = capture(model, ids, args.k, lens)
        keys = {
            "output-fixed": make_keys(states.h_out, "fixed", stats["output"]),
            "input-center": make_keys(states.h_in, "center"),
        }
        full_nll = states.final_nll.cpu()
        runs = {}
        for cfg in configs:
            runs[cfg["name"]] = {}
            for tau in args.taus:
                learn = cfg["memory"] is not None
                log = simulate(
                    replace(states, key=keys[cfg["key"]]),
                    readout,
                    build_memory(cfg, d_model, args).to(args.device),
                    StressGate(tau),
                    learn=learn,
                    audit_rate=cfg["audit"],
                    decay_per_token=learn,
                )
                runs[cfg["name"]][str(tau)] = run_metrics(log, args.k, n_layers, head, full_nll)
        valid = ~full_nll.isnan()
        results["texts"][text["name"]] = {"group": text["group"], "full_nll": full_nll[valid].mean().item(), "runs": runs}
        print(f"[{text['group']}] {text['name']}: pronto ({time.time() - start:.0f}s acumulados)", flush=True)

    print_summary(results, args.taus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp3_{args.model.split('/')[-1]}_k{args.k}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Resultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
