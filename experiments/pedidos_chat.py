"""Pedidos de chat do Experimento 7b, no estilo do Spec-Bench (Xia et al., 2024).

O Experimento 7 continuou documentos com o modelo base, e a geração gulosa
entrou em laço (77%–100% de 4-gramas repetidos). Aqui o modelo de instrução
responde a pedidos realistas, com e sem texto de apoio:
  - resumo e tradução de um trecho (a resposta pode copiar do trecho);
  - explicar código e editar código (documentar, que copia muito);
  - conversa e matemática, sem texto de apoio (copiar quase não ajuda).

Os trechos vêm dos mesmos arquivos congelados: validação de data/eval/ (sem o
Du Fu, que também está no teste) e teste de data/eval_dominio/. Os pedidos
livres foram escritos à mão, diferentes na validação e no teste.
"""

from __future__ import annotations

import torch

from textos import DOMAIN_DIR, EVAL_DIR

EXCERPT_TOKENS = 448

SUMMARY = {"pt": "Resuma o texto abaixo em um parágrafo.", "en": "Summarize the text below in one paragraph."}
TRANSLATE = "Translate the text below into English."
EXPLAIN = "Explain what the code below does, part by part."
DOCUMENT = "Add a docstring and type hints to every function in the code below. Return the complete code."

VALIDATION = [
    ("resumo", "Wikipédia: Fotossíntese", "pt_fotossintese.txt", SUMMARY["pt"]),
    ("resumo", "wikitext: Robert Boulter", "wikitext_robert_boulter.txt", SUMMARY["en"]),
    ("tradução", "Wikipédia: Revolução Industrial", "pt_revolucao_industrial.txt", TRANSLATE),
    ("código-explicar", "torch activation.py", "torch_activation.py", EXPLAIN),
    ("código-explicar", "test_fast_synapse.py", "test_fast_synapse.py", EXPLAIN),
    ("código-editar", "fast_synapse.py", "fast_synapse.py", DOCUMENT),
]
VALIDATION_FREE = [
    ("conversa", "dicas de Python", "What are three tips for writing clean Python code? Give a short example for each."),
    ("conversa", "céu azul", "Explique para uma criança de 10 anos por que o céu é azul."),
    ("conversa", "haicais", "Write a sequence of three haikus about autumn in a big city."),
    ("matemática", "retângulo", "A rectangle has a perimeter of 36 cm and its length is twice its width. What is its area? Think step by step."),
    ("matemática", "torneiras", "Se 3 torneiras iguais enchem um tanque em 12 horas, quanto tempo 4 torneiras levam? Pense passo a passo."),
]

TEST = [
    ("resumo", "Dom Casmurro", "pt_dom_casmurro.txt", SUMMARY["pt"]),
    ("resumo", "Braz Cubas", "pt_braz_cubas.txt", SUMMARY["pt"]),
    ("resumo", "wikitext: Du Fu", "wikitext_du_fu.txt", SUMMARY["en"]),
    ("resumo", "wikitext: Kiss You", "wikitext_kiss_you.txt", SUMMARY["en"]),
    ("resumo", "wikitext: Ise battleship", "wikitext_ise_battleship.txt", SUMMARY["en"]),
    ("tradução", "Quijote (es)", "es_quijote.txt", TRANSLATE),
    ("tradução", "Verwandlung (de)", "de_verwandlung.txt", TRANSLATE),
    ("tradução", "Tour du monde (fr)", "fr_tour_du_monde.txt", TRANSLATE),
    ("tradução", "especificação (MYELIN)", "pt_especificacao.md", TRANSLATE),
    ("código-explicar", "option.rs", "option.rs", EXPLAIN),
    ("código-explicar", "strings.go", "strings.go", EXPLAIN),
    ("código-explicar", "response.js", "response.js", EXPLAIN),
    ("código-editar", "numpy numeric.py", "numpy_numeric.py", DOCUMENT),
    ("código-editar", "jinja2 lexer.py", "jinja2_lexer.py", DOCUMENT),
    ("código-editar", "networkx weighted.py", "networkx_weighted.py", DOCUMENT),
]
TEST_FREE = [
    ("conversa", "e-mail de prazo", "Escreva um e-mail curto pedindo ao meu professor mais prazo para o trabalho final, explicando que fiquei doente."),
    ("conversa", "processo e thread", "Explain the difference between a process and a thread to a beginner programmer, using an analogy."),
    ("conversa", "plano de estudo", "Give me a 5-day study plan to learn the basics of linear algebra for machine learning."),
    ("conversa", "conto do farol", "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle."),
    ("conversa", "trabalho remoto", "Quais são as vantagens e desvantagens de trabalhar remotamente? Responda em tópicos."),
    ("matemática", "lápis", "A store sells pencils at 3 for $1.20. How much do 17 pencils cost? Think step by step."),
    ("matemática", "trens", "Um trem sai às 14h10 a 80 km/h; outro sai do mesmo ponto às 14h40 a 100 km/h, no mesmo sentido. A que horas o segundo alcança o primeiro? Pense passo a passo."),
    ("matemática", "máquinas", "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets? Explain your reasoning."),
    ("matemática", "maçãs", "Maria has twice as many apples as João. After she gives him 6 apples, they have the same amount. How many apples did Maria have at first? Solve step by step."),
]


def _prompt(tokenizer, request: str) -> torch.Tensor:
    messages = [{"role": "user", "content": request}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids


def _excerpt(tokenizer, path) -> str:
    ids = tokenizer(path.read_text(encoding="utf-8")).input_ids[:EXCERPT_TOKENS]
    return tokenizer.decode(ids)


def _requests(tokenizer, directory, with_text, free) -> list[dict]:
    requests = [
        {"name": name, "group": group, "ids": _prompt(tokenizer, f"{task}\n\n{_excerpt(tokenizer, directory / file)}")}
        for group, name, file, task in with_text
    ]
    requests += [{"name": name, "group": group, "ids": _prompt(tokenizer, request)} for group, name, request in free]
    return requests


def validation_requests(tokenizer) -> list[dict]:
    """11 pedidos de validação: {"name", "group", "ids"} com o template de chat do modelo."""
    return _requests(tokenizer, EVAL_DIR, VALIDATION, VALIDATION_FREE)


def test_requests(tokenizer) -> list[dict]:
    """24 pedidos de teste, em 6 grupos."""
    return _requests(tokenizer, DOMAIN_DIR, TEST, TEST_FREE)
