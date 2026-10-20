"""Textos de avaliação e de calibração compartilhados pelos experimentos.

Avaliação (nunca usados para treinar o tradutor nem para calibrar as chaves):
  - código-dev : 2 arquivos Python do próprio MYELIN (comentários em português);
                 usados no desenvolvimento, então valem só como referência;
  - pt         : README.md e docs/guia-de-estudo-e-pitch.md (prosa em português,
                 fora do treino do tradutor);
  - código-en  : um arquivo Python do PyTorch (comentários em inglês);
  - prosa-en   : artigos do split de TESTE do WikiText-2 (o tradutor viu o de treino).

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


def eval_texts(tokenizer, max_tokens: int = 1024, n_wikitext: int = 2) -> list[dict]:
    """Lista de {"name", "group", "ids"}; artigos do WikiText só entram se tiverem `max_tokens` tokens."""
    files = [
        ("código-dev", "src/myelin/fast_synapse.py", ROOT / "src" / "myelin" / "fast_synapse.py"),
        ("código-dev", "tests/test_fast_synapse.py", ROOT / "tests" / "test_fast_synapse.py"),
        ("pt", "README.md", ROOT / "README.md"),
        ("pt", "docs/guia-de-estudo-e-pitch.md", ROOT / "docs" / "guia-de-estudo-e-pitch.md"),
        ("código-en", "torch/nn/modules/activation.py", Path(torch.__file__).parent / "nn" / "modules" / "activation.py"),
    ]
    texts = []
    for group, name, path in files:
        texts.append({"name": name, "group": group, "ids": _ids(tokenizer, path.read_text(encoding="utf-8"), max_tokens)})

    for article in _wikitext_articles("test"):
        if len(texts) == len(files) + n_wikitext:
            break
        ids = _ids(tokenizer, article, max_tokens)
        if ids.shape[1] == max_tokens:
            title = article.strip().splitlines()[0].strip(" =")
            texts.append({"name": f"wikitext: {title}", "group": "prosa-en", "ids": ids})
    return texts


def calibration_ids(tokenizer, n_code: int = 4, n_prose: int = 4, max_tokens: int = 512, seed: int = 1):
    """Trechos da mesma distribuição do treino do tradutor, para as estatísticas fixas das chaves."""
    code_files = sorted(Path(transformers.__file__).parent.rglob("*.py"))
    random.Random(seed).shuffle(code_files)
    code = [f.read_text(encoding="utf-8", errors="ignore") for f in code_files]
    prose = _wikitext_articles("train")
    picked = [t for t in code if len(t) > 4 * max_tokens][:n_code] + [t for t in prose if len(t) > 4 * max_tokens][:n_prose]
    return [_ids(tokenizer, text, max_tokens) for text in picked]
