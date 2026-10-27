# Textos de avaliação: fontes e licenças

Cópias congeladas: editar o projeto não muda estes arquivos. O carregamento está em [`experiments/textos.py`](../../experiments/textos.py).

| arquivo | grupo | fonte | licença |
|---|---|---|---|
| `fast_synapse.py`, `test_fast_synapse.py` | código-dev | versões antigas de `src/myelin/fast_synapse.py` e `tests/test_fast_synapse.py` deste projeto | a do projeto |
| `torch_activation.py` | código-en | `torch/nn/modules/activation.py` (PyTorch) | BSD-3-Clause |
| `wikitext_robert_boulter.txt`, `wikitext_du_fu.txt` | prosa-en | split de teste do WikiText-2 (Merity et al., 2016) | CC BY-SA 3.0 |
| `pt_fotossintese.txt` | pt | Wikipédia em português, "Fotossíntese", revisão 73032270 (22/09/2026), introdução e primeiras seções | CC BY-SA 4.0 |
| `pt_revolucao_industrial.txt` | pt | Wikipédia em português, "Revolução Industrial", revisão 73117557 (03/10/2026), introdução e primeiras seções | CC BY-SA 4.0 |

## Mudança no grupo pt (27/10/2026)

Até o Experimento 7b, o grupo pt eram cópias do README e de um documento de apoio do projeto. Os dois artigos da Wikipédia acima tomaram o lugar deles, para que os dados de avaliação não dependam de documentos internos.

- **Os números publicados dos Experimentos 1 a 7b** usam os arquivos antigos, que continuam no histórico do git. Para reproduzi-los exatamente, use `git checkout db4b904 -- data/eval experiments/textos.py experiments/pedidos_chat.py`.
- **A partir do Experimento 8,** valem os textos desta tabela.
