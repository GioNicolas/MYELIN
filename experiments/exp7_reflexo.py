"""Experimento 7: o ciclo real da decodificação especulativa, com linhas de base justas e o reflexo do MYELIN.

Responde à revisão externa do Experimento 6:
  - simula o ciclo de verdade (rascunho → verificação → token bônus, com o
    professor chegando um ciclo depois), sobre texto GERADO pelo próprio modelo
    (guloso, a partir dos primeiros 256 tokens de cada documento), com o custo
    corrigido: verificação = 1 passo inteiro (ver myelin/speculative.py);
  - linhas de base: busca de n-gramas no contexto (Prompt Lookup Decoding) e
    cache neural justo (aprende o token do verificador, porta de familiaridade,
    λ ajustado, 1024 pares);
  - controles da memória de correções: pares embaralhados, correção média,
    última correção;
  - o REFLEXO: reconhecer a situação pelo estado oculto e repetir o que o
    verificador disse, sem rodar a cabeça (rascunho de graça).

Todos os hiperparâmetros (γ, n, τ, camada, λ, min_sim) são escolhidos nos 7
textos de validação (data/eval/) e só então medidos nos 15 de teste
(data/eval_dominio/).

    python experiments/exp7_reflexo.py --lens results/tuned_lens_Qwen2.5-0.5B_multi.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import KNNCache, NeuralCache, TunedLens, capture  # noqa: E402
from myelin.controls import LastCorrection, MeanCorrection, ShuffledKNN  # noqa: E402
from myelin.speculative import ChainDrafter, HeadDrafter, PromptLookupDrafter, ReflexDrafter, run_cycles  # noqa: E402
from myelin.wrapper import fit_key_stats, hf_readout, make_keys  # noqa: E402
from exp3_portao import head_cost_in_layers  # noqa: E402
from textos import calibration_ids, domain_texts, eval_texts  # noqa: E402

def grid(head_ks: list[int], reflex_layers: list[int]) -> dict[str, list[dict]]:
    """Famílias de rascunhadores e os hiperparâmetros que a validação escolhe."""
    families: dict[str, list[dict]] = {}
    for k in head_ks:
        families[f"estático k={k}"] = [{"kind": "head", "k": k, "gamma": g} for g in (1, 2, 3)]
        families[f"kNN 1024 k={k}"] = [
            {"kind": "head", "k": k, "gamma": g, "memory": "knn", "min_sim": s} for g in (1, 2, 3) for s in (0.5, 0.7)
        ]
        families[f"cache neural justo k={k}"] = [
            {"kind": "head", "k": k, "gamma": g, "cache_lambda": lam} for g in (1, 2, 3) for lam in (0.5, 0.9)
        ]
    families["busca de n-gramas (PLD)"] = [{"kind": "pld", "n_max": n, "gamma": g} for n in (2, 3, 4) for g in (3, 6)]
    families["reflexo MYELIN"] = [
        {"kind": "reflex", "layer": layer, "tau": tau, "gamma": g}
        for layer, tau, g in itertools.product(reflex_layers, (0.6, 0.75, 0.9), (3, 6))
    ]
    return families


def build(cfg: dict, bundle: dict, readout, head: float):
    kind = cfg["kind"]
    if kind == "head":
        states = bundle["states"][cfg["k"]]
        memory = cache = None
        if cfg.get("memory") == "knn":
            memory = KNNCache(capacity=1024, min_sim=cfg["min_sim"])
        elif cfg.get("memory") == "shuffled":
            memory = ShuffledKNN(capacity=1024, min_sim=cfg["min_sim"])
        elif cfg.get("memory") == "mean":
            memory = MeanCorrection()
        elif cfg.get("memory") == "last":
            memory = LastCorrection()
        if "cache_lambda" in cfg:
            cache = NeuralCache(lam=cfg["cache_lambda"], capacity=1024, min_sim=0.5)
        return HeadDrafter(states, readout, head, memory=memory, cache=cache)
    if kind == "pld":
        return PromptLookupDrafter(bundle["tokens"], n_max=cfg["n_max"])
    if kind == "reflex":
        return ReflexDrafter(bundle["keys"][cfg["layer"]], bundle["tokens"], tau=cfg["tau"])
    if kind == "chain":
        return ChainDrafter(build(cfg["first"], bundle, readout, head), build(cfg["second"], bundle, readout, head))
    raise ValueError(kind)


def run(cfg: dict, bundle: dict, readout, head: float, full_cost: float, extra: float) -> dict:
    drafter = build(cfg, bundle, readout, head)
    stats = run_cycles(drafter, bundle["final_top"], bundle["start"], bundle["end"], cfg["gamma"], full_cost, extra)
    return stats.as_dict()


@torch.no_grad()
def make_bundle(model, tokenizer, lens, text: dict, stats: dict, args) -> dict:
    """Gera a continuação gulosa e guarda o que os rascunhadores precisam."""
    prompt = text["ids"][:, : args.prompt_tokens].to(args.device)
    sequence = model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        do_sample=False,
        max_new_tokens=args.new_tokens,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    hidden = model(sequence, output_hidden_states=True)
    final_top = hidden.logits[0].argmax(-1)
    keys = {layer: make_keys(hidden.hidden_states[layer][0].float(), "fixed", stats[layer]) for layer in args.reflex_layers}
    del hidden
    states = {}
    for k in args.head_ks:
        s = capture(model, sequence, k, lens)
        states[k] = replace(s, key=make_keys(s.h_out, "fixed", stats[k]), h_in=None, h_out=None)

    tokens = sequence[0].tolist()
    start, end = prompt.shape[1] - 1, len(tokens) - 1
    generated = tokens[start + 1 :]
    seen, repeats = set(), 0
    for i in range(3, len(tokens)):
        gram = tuple(tokens[i - 3 : i + 1])
        if i > start and gram in seen:
            repeats += 1
        seen.add(gram)
    mismatch = sum(final_top[p].item() != tokens[p + 1] for p in range(start, end)) / max(1, end - start)
    return {
        "name": text["name"],
        "group": text["group"],
        "tokens": tokens,
        "final_top": final_top.tolist(),
        "start": start,
        "end": end,
        "keys": keys,
        "states": states,
        "repeat_4gram": repeats / max(1, len(generated)),
        "greedy_mismatch": mismatch,
        "preview": tokenizer.decode(generated[:60]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", type=Path, required=True, help="tradutor P_k (o amplo, corpus multi)")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=768)
    parser.add_argument("--head-ks", nargs="+", type=int, default=[8, 17], help="camadas do rascunho pela cabeça")
    parser.add_argument("--reflex-layers", nargs="+", type=int, default=[8, 17, 24], help="camadas da chave do reflexo")
    parser.add_argument("--extra-token-cost", type=float, default=0.02, help="custo de cada posição a mais na verificação")
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

    # 1. Validação: escolhe os hiperparâmetros de cada família.
    families = grid(args.head_ks, args.reflex_layers)
    validation = [make_bundle(model, tokenizer, lens, t, stats, args) for t in eval_texts(tokenizer, args.prompt_tokens + 64)]
    print(f"Validação: {len(validation)} textos gerados ({time.time() - start:.0f}s)", flush=True)
    chosen, val_scores = {}, {}
    for family, configs in families.items():
        scores = [np.mean([run(cfg, b, readout, head, full_cost, args.extra_token_cost)["speedup"] for b in validation]) for cfg in configs]
        best = int(np.argmax(scores))
        chosen[family], val_scores[family] = configs[best], float(scores[best])
        print(f"  {family:<28} melhor {configs[best]} → {scores[best]:.3f}× na validação", flush=True)
    del validation

    pld, reflex = chosen["busca de n-gramas (PLD)"], chosen["reflexo MYELIN"]
    chosen["PLD → reflexo"] = {"kind": "chain", "first": pld, "second": reflex, "gamma": max(pld["gamma"], reflex["gamma"])}
    best_knn = max((f for f in chosen if f.startswith("kNN")), key=lambda f: val_scores[f])
    controls = {
        f"controle: {name} ({best_knn.split()[-1]})": {**chosen[best_knn], "memory": memory}
        for name, memory in (("pares embaralhados", "shuffled"), ("correção média", "mean"), ("última correção", "last"))
    }

    # 2. Teste: as escolhas da validação nos 15 textos de domínio.
    configs = {**chosen, **controls}
    results = {"config": vars(args), "chosen": chosen, "validation_speedup": val_scores, "texts": {}}
    for text in domain_texts(tokenizer, args.prompt_tokens + 64):
        bundle = make_bundle(model, tokenizer, lens, text, stats, args)
        runs = {name: run(cfg, bundle, readout, head, full_cost, args.extra_token_cost) for name, cfg in configs.items()}
        results["texts"][text["name"]] = {
            "group": text["group"],
            "repeat_4gram": bundle["repeat_4gram"],
            "greedy_mismatch": bundle["greedy_mismatch"],
            "preview": bundle["preview"],
            "runs": runs,
        }
        print(f"[{text['group']}] {text['name']}: repetição de 4-gramas {bundle['repeat_4gram']:.0%} ({time.time() - start:.0f}s)", flush=True)
        del bundle

    print_summary(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = args.out_dir / f"exp7_{args.model.split('/')[-1]}{tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


def print_summary(results: dict) -> None:
    texts = list(results["texts"].values())
    groups = list(dict.fromkeys(t["group"] for t in texts))
    names = list(texts[0]["runs"])

    def mean(name: str, metric: str, group: str | None = None) -> float:
        chosen = [t for t in texts if group is None or t["group"] == group]
        return float(np.mean([t["runs"][name][metric] for t in chosen]))

    print("\nSpeedup sobre a decodificação normal (média dos textos; > 1 = mais rápido), texto gerado pelo modelo")
    print(f"{'rascunhador':<34} " + " ".join(f"{g:>12}" for g in groups) + f" {'todos':>7} {'aceit.':>7} {'tok/ciclo':>9} {'propõe':>7}")
    for name in names:
        print(
            f"{name:<34} " + " ".join(f"{mean(name, 'speedup', g):>11.2f}×" for g in groups)
            + f" {mean(name, 'speedup'):>6.2f}× {mean(name, 'acceptance'):>7.0%}"
            + f" {mean(name, 'tokens_per_cycle'):>9.2f} {mean(name, 'proposal_rate'):>7.0%}"
        )
    print("\nRepetição de 4-gramas no texto gerado (laços da geração gulosa favorecem a busca de n-gramas e o reflexo):")
    print("  " + ", ".join(f"{g} {np.mean([t['repeat_4gram'] for t in texts if t['group'] == g]):.0%}" for g in groups))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
