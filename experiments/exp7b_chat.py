"""Experimento 7b: o reflexo em texto com pouca repetição (modelo de instrução, pedidos de chat).

No Experimento 7, o texto gerado pelo modelo base era quase todo laço, um
regime em que copiar do contexto é fácil. Aqui o mesmo ciclo exato da
auto-especulação gulosa roda sobre respostas do Qwen2.5-0.5B-Instruct a 24
pedidos de chat em 6 grupos (ver pedidos_chat.py): resumo, tradução, explicar
código, editar código, conversa e matemática.

Novidades em relação ao Experimento 7:
  - variante do reflexo que exige 2 tokens iguais (min_match = 2) e a chave
    também na camada 12;
  - o custo da busca do reflexo entra na conta (um produto escalar por
    candidato);
  - correntes, cada elo com o seu γ: reflexo → PLD, PLD → reflexo, e a cabeça
    com memória quando os de cópia se calam.

Hiperparâmetros escolhidos nos 11 pedidos de validação, medidos nos 24 de teste.

    python experiments/exp7b_chat.py --lens results/tuned_lens_Qwen2.5-0.5B-Instruct_multi.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import TunedLens  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout  # noqa: E402
from exp3_portao import head_cost_in_layers  # noqa: E402
from exp7_reflexo import chain, generate_bundle, grid, print_summary, run  # noqa: E402
from pedidos_chat import test_requests, validation_requests  # noqa: E402
from textos import calibration_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k amplo, treinado para ESTE modelo")
    parser.add_argument("--new-tokens", type=int, default=512, help="limite da resposta (para antes no fim da resposta)")
    parser.add_argument("--head-ks", nargs="+", type=int, default=[8, 17])
    parser.add_argument("--reflex-layers", nargs="+", type=int, default=[8, 12, 17, 24])
    parser.add_argument("--extra-token-cost", type=float, default=0.02)
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    lens = TunedLens.load(args.lens, map_location=args.device).eval()
    n_layers = model.config.num_hidden_layers
    head = head_cost_in_layers(model)
    full_cost = n_layers + head
    readout = hf_readout(model)
    calibration = [ids.to(args.device) for ids in calibration_ids(tokenizer)]
    stats = {layer: fit_key_stats(model, calibration, layer)["output"] for layer in sorted({*args.head_ks, *args.reflex_layers})}
    print(f"Modelo {args.model}: {n_layers} camadas, cabeça ≈ {head:.1f}, passo completo = {full_cost:.1f} camadas")
    start = time.time()

    def bundle_of(request: dict) -> dict:
        return generate_bundle(model, tokenizer, lens, request["ids"], request["name"], request["group"], stats, args)

    def score(cfg: dict, bundles: list[dict]) -> float:
        return float(np.mean([run(cfg, b, readout, head, full_cost, args.extra_token_cost)["speedup"] for b in bundles]))

    # 1. Validação: escolhe os hiperparâmetros de cada família.
    validation = [bundle_of(r) for r in validation_requests(tokenizer)]
    print(f"Validação: {len(validation)} respostas geradas ({time.time() - start:.0f}s)", flush=True)
    chosen, val_scores = {}, {}
    for family, configs in grid(args.head_ks, args.reflex_layers, reflex_matches=(1, 2)).items():
        scores = [score(cfg, validation) for cfg in configs]
        best = int(np.argmax(scores))
        chosen[family], val_scores[family] = configs[best], scores[best]
        print(f"  {family:<28} melhor {configs[best]} → {scores[best]:.3f}× na validação", flush=True)

    # Correntes das escolhas (sem ajuste próprio): os de cópia primeiro, a cabeça com memória por último.
    pld, reflex = chosen["busca de n-gramas (PLD)"], chosen["reflexo MYELIN"]
    best_knn = max((f for f in chosen if f.startswith("kNN")), key=lambda f: val_scores[f])
    knn = chosen[best_knn]
    chains = {
        "reflexo → PLD": chain(reflex, pld),
        "PLD → reflexo": chain(pld, reflex),
        f"PLD → {best_knn}": chain(pld, knn),
        f"reflexo → {best_knn}": chain(reflex, knn),
        f"reflexo → PLD → {best_knn}": chain(reflex, pld, knn),
    }
    for name, cfg in chains.items():
        val_scores[name] = score(cfg, validation)
        print(f"  {name:<28} → {val_scores[name]:.3f}× na validação", flush=True)
    chosen.update(chains)
    del validation
    controls = {
        f"controle: {name} ({best_knn.split()[-1]})": {**knn, "memory": memory}
        for name, memory in (("pares embaralhados", "shuffled"), ("correção média", "mean"), ("última correção", "last"))
    }

    # 2. Teste.
    configs = {**chosen, **controls}
    results = {"config": vars(args), "chosen": chosen, "validation_speedup": val_scores, "texts": {}}
    for request in test_requests(tokenizer):
        bundle = bundle_of(request)
        runs = {name: run(cfg, bundle, readout, head, full_cost, args.extra_token_cost) for name, cfg in configs.items()}
        results["texts"][request["name"]] = {
            "group": request["group"],
            "prompt_tokens": bundle["start"] + 1,
            "answer_tokens": bundle["end"] - bundle["start"],
            "repeat_4gram": bundle["repeat_4gram"],
            "greedy_mismatch": bundle["greedy_mismatch"],
            "preview": bundle["preview"],
            "runs": runs,
        }
        print(
            f"[{request['group']}] {request['name']}: {bundle['end'] - bundle['start']} tokens de resposta, "
            f"repetição de 4-gramas {bundle['repeat_4gram']:.0%} ({time.time() - start:.0f}s)",
            flush=True,
        )
        del bundle

    print_summary(results)
    print_head_to_head(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp7b_{Path(args.model).name}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


def print_head_to_head(results: dict) -> None:
    texts = list(results["texts"].values())
    wins = sum(t["runs"]["reflexo MYELIN"]["speedup"] > t["runs"]["busca de n-gramas (PLD)"]["speedup"] for t in texts)
    print(f"\nReflexo > busca de n-gramas em {wins} de {len(texts)} respostas")
    print("Speedup no total de tokens (respostas longas pesam mais):")
    for name in texts[0]["runs"]:
        tokens = sum(t["runs"][name]["tokens"] for t in texts)
        cost = sum(t["runs"][name]["tokens"] / t["runs"][name]["speedup"] for t in texts)
        print(f"  {name:<40} {tokens / cost:.2f}×")
    lengths = [t["answer_tokens"] for t in texts]
    print(f"Tamanho das respostas: mediana {np.median(lengths):.0f} tokens (de {min(lengths)} a {max(lengths)})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
