"""Experimento 0: o quanto cada camada de um LLM real já "sabe" a resposta final?

Para cada camada ℓ, lê o estado oculto pela cabeça do próprio modelo
(logit lens: lm_head(norm(h_ℓ))) e mede, token a token:

  - S       : estresse (entropia normalizada) da previsão da camada ℓ
  - concorda: top-1 da camada ℓ == top-1 da camada final
  - KL      : D_KL(p_L || p_ℓ)
  - S<=τ    : fração de tokens que sairiam cedo num Early-Exit Estático
  - precisão: entre os que sairiam cedo, quantos concordam com a final
              (calibração do portão: confiante e errado é o pior caso)

Compara a 1ª e a 2ª metade de cada texto: se a camada rasa melhora na 2ª
metade SEM sinapse, isso é o efeito do contexto (atenção copiando padrões),
a linha de base que o MYELIN precisa superar no H1.

    python experiments/exp0_profundidade_por_camada.py   # ou pelo notebook no Colab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import entropy_stress  # noqa: E402

DEFAULT_TEXTS = [
    ROOT / "src" / "myelin" / "fast_synapse.py",
    ROOT / "tests" / "test_fast_synapse.py",
    ROOT / "README.md",
]


@torch.no_grad()
def layer_logits(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Logits de cada camada 1..L para a sequência, lidos pela cabeça final."""
    out = model(input_ids, output_hidden_states=True)
    norm, head = model.model.norm, model.lm_head
    # hidden_states[0] é o embedding; o último já é a saída final, usamos out.logits.
    per_layer = [head(norm(h))[0] for h in out.hidden_states[1:-1]]
    return per_layer + [out.logits[0]]


def summarize(logits: list[torch.Tensor], tau: float) -> list[dict[str, float]]:
    final = logits[-1].float()
    final_top = final.argmax(-1)
    log_p_final = F.log_softmax(final, -1)
    rows = []
    for layer, lg in enumerate(logits, start=1):
        lg = lg.float()
        stress = entropy_stress(lg)
        agree = lg.argmax(-1) == final_top
        exits = stress <= tau
        kl = F.kl_div(F.log_softmax(lg, -1), log_p_final, log_target=True, reduction="none").sum(-1)
        rows.append(
            {
                "layer": layer,
                "stress": stress.mean().item(),
                "agree": agree.float().mean().item(),
                "kl": kl.mean().item(),
                "exit_rate": exits.float().mean().item(),
                "exit_precision": agree[exits].float().mean().item() if exits.any() else float("nan"),
            }
        )
    return rows


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
            f"{ra['exit_precision']:>6.0%} → {rb['exit_precision']:<6.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--texts", nargs="*", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"Modelo {args.model}: {n_layers} camadas, τ={args.tau}, device={args.device}")

    results = {}
    for path in args.texts:
        ids = tokenizer(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids
        ids = ids[:, : args.max_tokens].to(args.device)
        logits = layer_logits(model, ids)
        mid = ids.shape[1] // 2
        halves = {
            "1a metade": summarize([lg[:mid] for lg in logits], args.tau),
            "2a metade": summarize([lg[mid:] for lg in logits], args.tau),
        }
        print_table(f"{path.relative_to(ROOT)} ({ids.shape[1]} tokens)", halves)
        results[str(path.relative_to(ROOT))] = halves

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"exp0_{args.model.split('/')[-1]}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
