"""Experimento 0: o quanto cada camada de um LLM real já "sabe" a resposta final?

Para cada camada ℓ, lê o estado oculto pela cabeça do próprio modelo
(logit lens: lm_head(norm(h_ℓ))) ou, com --lens, pelo tradutor treinado
(tuned lens: lm_head(norm(P_ℓ(h_ℓ))), ver exp0b_tuned_lens.py) e mede, token a token:

  - S       : estresse (entropia normalizada) da previsão da camada ℓ
  - concorda: top-1 da camada ℓ == top-1 da camada final
  - KL      : D_KL(p_L || p_ℓ)
  - S<=τ    : fração de tokens que sairiam cedo num Early-Exit Estático
  - precisão: entre os que sairiam cedo, quantos concordam com a final
              (calibração do portão: confiante e errado é o pior caso)

Compara a 1ª e a 2ª metade de cada texto: se a camada rasa melhora na 2ª
metade SEM sinapse, isso é o efeito do contexto (atenção copiando padrões),
a linha de base que o MYELIN precisa superar no H1.

    python experiments/exp0_profundidade_por_camada.py                 # logit lens
    python experiments/exp0_profundidade_por_camada.py --lens P.pt     # tuned lens
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import TunedLens, entropy_stress  # noqa: E402

DEFAULT_TEXTS = [
    ROOT / "src" / "myelin" / "fast_synapse.py",
    ROOT / "tests" / "test_fast_synapse.py",
    ROOT / "README.md",
]


@torch.no_grad()
def layer_stats(
    model, input_ids: torch.Tensor, tau: float, lens: TunedLens | None = None
) -> dict[str, list[dict[str, float]]]:
    """Estatísticas de cada camada 1..L, separadas na 1ª e na 2ª metade do texto.

    Os logits de uma camada ocupam tokens × |V| (~600 MB para 1024 tokens no
    Qwen), então cada camada é lida pela cabeça, resumida e descartada.
    """
    out = model(input_ids, output_hidden_states=True)
    norm, head = model.model.norm, model.lm_head
    # hidden_states[0] é o embedding; o último já é a saída final, usamos out.logits.
    states = out.hidden_states[1:-1]
    n_tokens = input_ids.shape[1]
    halves = {"1a metade": slice(0, n_tokens // 2), "2a metade": slice(n_tokens // 2, n_tokens)}

    results = {}
    for name, span in halves.items():
        final = out.logits[0, span].float()
        final_top = final.argmax(-1)
        log_p_final = F.log_softmax(final, -1)
        rows = []
        for layer in range(1, len(states) + 2):
            if layer > len(states):
                lg = final
            else:
                h = states[layer - 1][0, span]
                lg = head(norm(h if lens is None else lens(layer, h))).float()
            stress = entropy_stress(lg)
            agree = lg.argmax(-1) == final_top
            exits = stress <= tau
            kl = F.kl_div(F.log_softmax(lg, -1), log_p_final, log_target=True, reduction="none")
            rows.append(
                {
                    "layer": layer,
                    "stress": stress.mean().item(),
                    "agree": agree.float().mean().item(),
                    "kl": kl.sum(-1).mean().item(),
                    "exit_rate": exits.float().mean().item(),
                    "exit_precision": agree[exits].float().mean().item() if exits.any() else float("nan"),
                }
            )
            del lg, kl
        results[name] = rows
    return results


def _pct(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.0%}"


def print_table(title: str, halves: dict[str, list[dict[str, float]]]) -> None:
    a, b = halves["1a metade"], halves["2a metade"]
    print(f"\n{title}")
    print(
        f"{'camada':>6} | {'S médio':>7} | {'concorda c/ final':>17} | {'KL':>6} | "
        f"{'S<=τ':>11} | {'precisão se S<=τ':>16}"
    )
    print(f"{'':>6} | {'':>7} | {'1ª → 2ª metade':>17} | {'':>6} | {'1ª → 2ª':>11} | {'1ª → 2ª':>16}")
    for ra, rb in zip(a, b):
        print(
            f"{ra['layer']:>6} | {(ra['stress'] + rb['stress']) / 2:>7.2f} | "
            f"{ra['agree']:>7.0%} → {rb['agree']:<7.0%} | {(ra['kl'] + rb['kl']) / 2:>6.2f} | "
            f"{ra['exit_rate']:>4.0%} → {rb['exit_rate']:<4.0%} | "
            f"{_pct(ra['exit_precision']):>6} → {_pct(rb['exit_precision']):<6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--texts", nargs="*", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lens", type=Path, help="tradutor P_k treinado (tuned lens); sem ele, logit lens")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    lens = TunedLens.load(args.lens, map_location=args.device).eval() if args.lens else None
    reader = "tuned lens" if lens else "logit lens"
    print(f"Modelo {args.model}: {n_layers} camadas, τ={args.tau}, leitura={reader}, device={args.device}")

    results = {}
    for path in args.texts:
        ids = tokenizer(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids
        ids = ids[:, : args.max_tokens].to(args.device)
        halves = layer_stats(model, ids, args.tau, lens)
        print_table(f"{path.relative_to(ROOT)} ({ids.shape[1]} tokens)", halves)
        results[str(path.relative_to(ROOT))] = halves

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tunedlens" if lens else ""
    out_file = args.out_dir / f"exp0_{args.model.split('/')[-1]}{suffix}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
