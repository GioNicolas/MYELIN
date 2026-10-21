# Textos do Experimento 4 (adaptação ao domínio)

Trechos congelados em 21/10/2026, de ~16 mil caracteres cada (3.000 a 5.000 tokens do Qwen2.5). O experimento lê os primeiros 2048 tokens.
Nenhum foi usado no treino do tradutor $P_k$, que viu só prosa em inglês (WikiText-2, split de treino) e código Python da biblioteca `transformers`.

| grupo | arquivo | fonte | licença |
|---|---|---|---|
| pt (fora do tradutor) | `pt_dom_casmurro.txt` | Machado de Assis, *Dom Casmurro*, Projeto Gutenberg #55752, a partir do cap. I | domínio público |
| | `pt_braz_cubas.txt` | Machado de Assis, *Memorias Posthumas de Braz Cubas*, Gutenberg #54829, cap. I (ortografia antiga) | domínio público |
| | `pt_especificacao.md` | `docs/especificacao-matematica-e-experimento.md` deste repositório (português técnico moderno) | do autor |
| outros idiomas (fora) | `es_quijote.txt` | Cervantes, *Don Quijote*, Gutenberg #2000, cap. I | domínio público |
| | `de_verwandlung.txt` | Kafka, *Die Verwandlung*, Gutenberg #22367, cap. I | domínio público |
| | `fr_tour_du_monde.txt` | Verne, *Le tour du monde en quatre-vingts jours*, Gutenberg #800, cap. I | domínio público |
| código não-Python (fora) | `option.rs` | `rust-lang/rust`, `library/core/src/option.rs` (branch master), a partir de `impl<T> Option<T>` | MIT / Apache-2.0 |
| | `strings.go` | `golang/go`, `src/strings/strings.go` (master) | BSD-3-Clause |
| | `response.js` | `expressjs/express`, `lib/response.js` (master) | MIT |
| Python (controle, dentro) | `numpy_numeric.py` | numpy 2.5.4, `numpy/_core/numeric.py` | BSD-3-Clause |
| | `jinja2_lexer.py` | jinja2 3.1.6, `jinja2/lexer.py` | BSD-3-Clause |
| | `networkx_weighted.py` | networkx 3.6.1, `networkx/algorithms/shortest_paths/weighted.py` | BSD-3-Clause |
| prosa-en (controle, dentro) | `wikitext_du_fu.txt` | WikiText-2, split de teste, artigo *Du Fu* | CC BY-SA |
| | `wikitext_kiss_you.txt` | WikiText-2, teste, *Kiss You (One Direction song)* | CC BY-SA |
| | `wikitext_ise_battleship.txt` | WikiText-2, teste, *Ise-class battleship* | CC BY-SA |
