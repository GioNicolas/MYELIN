"""Experimento 0b: treina o tradutor P_k (tuned lens) de cada camada do Qwen.

O modelo fica 100% congelado. Para cada camada ℓ = 1..L-1, uma transformação
afim P_ℓ aprende a fazer lm_head(norm(P_ℓ(h_ℓ))) imitar a distribuição final,
minimizando D_KL(p_L || p_ℓ). Depois, o Experimento 0 roda de novo com
--lens para ver em que camada a previsão traduzida passa a concordar com a
final e se o estresse fica calibrado.

Corpus de treino (nunca inclui os textos de avaliação):
  - "en-py" (padrão, Experimento 0b): metade prosa em inglês (WikiText-2) e
    metade código Python (arquivos da própria biblioteca transformers).
  - "multi" (controle do Experimento 4): o mesmo en-py em metade dos tokens,
    mais Wikipedia em pt/es/de/fr e código Rust/Go/JavaScript de projetos que
    NÃO estão na avaliação. Pergunta: com um tradutor amplo, o ganho do MYELIN
    fora do domínio sobrevive, ou era só compensação de um tradutor estreito?

    python experiments/exp0b_tuned_lens.py --model Qwen/Qwen2.5-0.5B --out-dir results
    python experiments/exp0b_tuned_lens.py --corpus multi --out-dir results
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myelin import TunedLens  # noqa: E402


# Projetos de código do corpus "multi": licenças permissivas e nenhum deles na
# avaliação (lá estão rust-lang/rust, golang/go e expressjs/express).
CODE_REPOS = {
    ".rs": ["BurntSushi/ripgrep", "serde-rs/serde", "tokio-rs/tokio"],
    ".go": ["spf13/cobra", "gin-gonic/gin", "urfave/cli"],
    ".js": ["axios/axios", "jquery/jquery", "moment/moment"],
}
WIKI_LANGS = ["pt", "es", "de", "fr"]


def _take(tokenizer, documents: Iterator[str], budget: int) -> list[int]:
    """Tokeniza documentos (separados por EOS) até juntar `budget` tokens."""
    ids: list[int] = []
    for doc in documents:
        ids += tokenizer(doc).input_ids + [tokenizer.eos_token_id]
        if len(ids) >= budget:
            break
    return ids[:budget]


def _en_py_documents(seed: int) -> tuple[list[str], Iterator[str]]:
    from datasets import load_dataset

    lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
    prose = ["".join(lines[i : i + 200]) for i in range(0, len(lines), 200)]
    code_files = sorted(Path(transformers.__file__).parent.rglob("*.py"))
    random.Random(seed).shuffle(code_files)
    return prose, (path.read_text(encoding="utf-8", errors="ignore") for path in code_files)


def build_corpus(tokenizer, max_tokens: int, seed: int = 0) -> torch.Tensor:
    """Fluxo 1-D de tokens: metade WikiText-2 (prosa), metade código Python."""
    prose, code = _en_py_documents(seed)
    prose_ids = _take(tokenizer, iter(prose), max_tokens // 2)
    code_ids = _take(tokenizer, code, max_tokens - len(prose_ids))
    return torch.tensor(prose_ids + code_ids)


def _wikipedia(lang: str, min_chars: int = 2000) -> Iterator[str]:
    from datasets import load_dataset

    for article in load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True):
        if len(article["text"]) >= min_chars:
            yield article["text"]


def _repo_files(extension: str, cache: Path, seed: int) -> Iterator[str]:
    files = []
    for repo in CODE_REPOS[extension]:
        target = cache / repo.replace("/", "__")
        if not target.exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{repo}.git", str(target)], check=True)
        files += sorted(path for path in target.rglob(f"*{extension}") if ".git" not in path.parts)
    random.Random(seed).shuffle(files)
    return (path.read_text(encoding="utf-8", errors="ignore") for path in files)


def build_multi_corpus(tokenizer, max_tokens: int, seed: int = 0, cache: Path = ROOT / ".cache" / "code") -> torch.Tensor:
    """Metade en-py (como build_corpus); a outra metade dividida entre pt/es/de/fr e Rust/Go/JS."""
    cache.mkdir(parents=True, exist_ok=True)
    prose, code = _en_py_documents(seed)
    ids = _take(tokenizer, iter(prose), max_tokens // 4) + _take(tokenizer, code, max_tokens // 4)
    sources = [_wikipedia(lang) for lang in WIKI_LANGS] + [_repo_files(ext, cache, seed) for ext in CODE_REPOS]
    for source in sources:
        ids += _take(tokenizer, source, max_tokens // (2 * len(sources)))
    return torch.tensor(ids)


def batch_stream(
    tokens: torch.Tensor, seq_len: int, batch_size: int, seed: int = 0
) -> Iterator[torch.Tensor]:
    """Sorteia, sem parar, lotes de trechos (batch_size, seq_len) do corpus."""
    chunks = tokens[: len(tokens) // seq_len * seq_len].view(-1, seq_len)
    g = torch.Generator().manual_seed(seed)
    while True:
        for idx in torch.randperm(len(chunks), generator=g).split(batch_size):
            if len(idx) == batch_size:
                yield chunks[idx]


def train_lens(
    model,
    lens: TunedLens,
    batches: Iterator[torch.Tensor],
    steps: int,
    lr: float = 1e-3,
    loss_positions: int = 512,
    device: str = "cpu",
    log_every: int = 50,
) -> list[list[float]]:
    """Treina todos os tradutores juntos; retorna o KL médio por camada a cada log."""
    norm, head = model.model.norm, model.lm_head
    optimizer = torch.optim.AdamW(lens.parameters(), lr=lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    g = torch.Generator().manual_seed(0)
    history, running, start = [], torch.zeros(lens.n_layers - 1), time.time()

    for step in range(1, steps + 1):
        ids = next(batches).to(device)
        with torch.no_grad():
            out = model.model(ids, output_hidden_states=True)
            # Logits de todos os tokens não cabem na memória (tokens × |V|):
            # o KL é medido numa amostra de posições a cada passo.
            pos = torch.randperm(ids.numel(), generator=g)[:loss_positions].to(device)
            final = head(out.last_hidden_state.reshape(-1, lens.d_model)[pos]).float()
            log_p_final = F.log_softmax(final, -1)
            # hidden_states[0] é o embedding; [1..L-1] são as saídas das camadas 1..L-1.
            states = [h.reshape(-1, lens.d_model)[pos] for h in out.hidden_states[1:-1]]
            del out, final

        optimizer.zero_grad(set_to_none=True)
        for layer, h in enumerate(states, start=1):
            log_p = F.log_softmax(head(norm(lens(layer, h))).float(), -1)
            loss = F.kl_div(log_p, log_p_final, log_target=True, reduction="batchmean")
            loss.backward()  # camada a camada: só um grafo de |V| logits na memória
            running[layer - 1] += loss.item()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            kl = (running / n).tolist()
            history.append(kl)
            running.zero_()
            quarter = [kl[i] for i in (len(kl) // 4, len(kl) // 2, 3 * len(kl) // 4)]
            print(
                f"passo {step:>5}/{steps} | KL médio {sum(kl) / len(kl):6.2f} | "
                f"camadas L/4, L/2, 3L/4: {quarter[0]:5.2f} {quarter[1]:5.2f} {quarter[2]:5.2f} | "
                f"{time.time() - start:5.0f}s",
                flush=True,
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--loss-positions", type=int, default=512)
    parser.add_argument("--corpus-tokens", type=int, default=3_000_000)
    parser.add_argument("--corpus", choices=["en-py", "multi"], default="en-py")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.device == "cpu":
        print("Aviso: sem GPU o treino pode levar horas; prefira GPU (Colab T4 ou Kaggle).")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    model.requires_grad_(False)
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    print(f"Modelo {args.model}: {n_layers} camadas, d={d_model}, device={args.device}")

    if args.corpus == "multi":
        corpus = build_multi_corpus(tokenizer, args.corpus_tokens)
        print(f"Corpus: {len(corpus):,} tokens (1/4 WikiText-2, 1/4 Python, 1/2 Wikipedia pt/es/de/fr e Rust/Go/JS)")
    else:
        corpus = build_corpus(tokenizer, args.corpus_tokens)
        print(f"Corpus: {len(corpus):,} tokens (metade WikiText-2, metade código Python)")

    lens = TunedLens(n_layers, d_model).to(args.device)
    history = train_lens(
        model,
        lens,
        batch_stream(corpus, args.seq_len, args.batch_size),
        steps=args.steps,
        lr=args.lr,
        loss_positions=args.loss_positions,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_multi" if args.corpus == "multi" else ""
    out_file = args.out_dir / f"tuned_lens_{args.model.split('/')[-1]}{suffix}.pt"
    lens.save(out_file, model=args.model, steps=args.steps, corpus=args.corpus, kl_history=history)
    print(f"\nKL final por camada: {' '.join(f'{kl:.2f}' for kl in history[-1])}")
    print(f"Tradutor salvo em {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
