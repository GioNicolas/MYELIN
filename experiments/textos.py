"""Textos de avaliação e de calibração compartilhados pelos experimentos.

Avaliação: cópias CONGELADAS em data/eval/ (editar o projeto não muda os
números; até o Experimento 2 eles eram lidos dos arquivos vivos, e o README
mudou entre rodadas). Nenhum foi usado para treinar o tradutor nem para
calibrar as chaves:
  - código-dev : 2 arquivos Python do próprio MYELIN (comentários em português);
                 usados no desenvolvimento, então valem só como referência;
  - pt         : README.md e docs/guia-de-estudo-e-pitch.md (prosa em português,
                 fora do treino do tradutor);
  - código-en  : torch/nn/modules/activation.py (comentários em inglês);
  - prosa-en   : 2 artigos do split de TESTE do WikiText-2 (o tradutor viu o de treino).

Calibração (mesma distribuição do treino do tradutor): arquivos Python da
biblioteca transformers e artigos do split de treino do WikiText-2.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]


def _wikitext_articles(split: str) -> list[str]:
    from datasets import load_dataset

    articles, current = [], []
    for line in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]:
        if line.startswith(" = ") and not line.startswith(" = = ") and current:
            articles.append("".join(current))
            current = []
        current.append(line)
    if current:
        articles.append("".join(current))
    return articles


def _ids(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens]


EVAL_DIR = ROOT / "data" / "eval"
EVAL_FILES = [
    ("código-dev", "src/myelin/fast_synapse.py", "fast_synapse.py"),
    ("código-dev", "tests/test_fast_synapse.py", "test_fast_synapse.py"),
    ("pt", "README.md", "README.md"),
    ("pt", "docs/guia-de-estudo-e-pitch.md", "guia-de-estudo-e-pitch.md"),
    ("código-en", "torch/nn/modules/activation.py", "torch_activation.py"),
    ("prosa-en", "wikitext: Robert Boulter", "wikitext_robert_boulter.txt"),
    ("prosa-en", "wikitext: Du Fu", "wikitext_du_fu.txt"),
]


def eval_texts(tokenizer, max_tokens: int = 1024) -> list[dict]:
    """Lista de {"name", "group", "ids"} a partir das cópias congeladas em data/eval/."""
    return [
        {"name": name, "group": group, "ids": _ids(tokenizer, (EVAL_DIR / file).read_text(encoding="utf-8"), max_tokens)}
        for group, name, file in EVAL_FILES
    ]


def calibration_ids(tokenizer, n_code: int = 4, n_prose: int = 4, max_tokens: int = 512, seed: int = 1):
    """Trechos da mesma distribuição do treino do tradutor, para as estatísticas fixas das chaves."""
    code_files = sorted(Path(transformers.__file__).parent.rglob("*.py"))
    random.Random(seed).shuffle(code_files)
    code = [f.read_text(encoding="utf-8", errors="ignore") for f in code_files]
    prose = _wikitext_articles("train")
    picked = [t for t in code if len(t) > 4 * max_tokens][:n_code] + [t for t in prose if len(t) > 4 * max_tokens][:n_prose]
    return [_ids(tokenizer, text, max_tokens) for text in picked]
