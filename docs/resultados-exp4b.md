# Resultados — Experimento 4b: o ganho sobrevive a um tradutor amplo?

**Data:** 22 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B`, pesos congelados, GPU T4 do Kaggle  
**O que muda em relação ao Experimento 4:** só o tradutor $P_{17}$. O tradutor estreito usava WikiText-2 e Python. O tradutor **amplo** ("multi") usa:
- 1/4 WikiText-2 e 1/4 Python;
- 1/2 dividida entre Wikipedia em pt/es/de/fr e código Rust/Go/JS de 9 projetos que **não** estão na avaliação.

Mesmos hiperparâmetros (500 passos) e os mesmos 15 textos, configurações e portão do Experimento 4.

**Pergunta:** o ganho do MYELIN fora do domínio era adaptação ao texto, ou só compensação de um tradutor estreito?

**Scripts:**
- [`experiments/exp0b_tuned_lens.py --corpus multi`](../experiments/exp0b_tuned_lens.py);
- [`experiments/exp4_comparar.py`](../experiments/exp4_comparar.py);
- notebook [`notebooks/exp4b_tradutor_amplo.ipynb`](../notebooks/exp4b_tradutor_amplo.ipynb).

---

## As quatro comparações (mesma taxa de saída, nats; negativo = a 1ª coluna é melhor)

| comparação | pt | idiomas | código-outro | Python | prosa-en |
|---|---|---|---|---|---|
| **1. estático amplo − estático estreito** (o tradutor amplo, sozinho) | −0,98 | −1,17 | −0,77 | −0,02 | +0,07 |
| **2. viés + kNN estreito − estático amplo** (adaptar lendo × treinar antes) | +0,06 | +0,04 | +0,04 | −0,21 | −0,03 |
| **3a. kNN 256 amplo − estático amplo** (o ganho que sobra) | **−0,14** | **−0,14** | **−0,13** | **−0,20** | −0,03 |
| 3b. viés + kNN amplo − estático amplo | −0,17 | −0,13 | −0,11 | −0,20 | +0,01 |
| 3c. viés lento amplo − estático amplo | −0,07 | −0,14 | +0,01 | −0,06 | +0,04 |
| 3d. cache neural amplo − estático amplo | −0,05 | −0,04 | −0,11 | −0,07 | −0,03 |
| **4. kNN 256 amplo − estático estreito** (os dois juntos) | −0,98 | −1,31 | −0,97 | −0,21 | +0,04 |

**Por texto, kNN 256 com o tradutor amplo:**
- melhora em 10 dos 12 textos que não são prosa inglesa (de −0,07 a −0,39; os maiores: `jinja2_lexer.py` −0,39, `response.js` −0,38, especificação −0,29);
- empata em `option.rs` (+0,03) e `strings.go` (−0,03);
- prosa inglesa: −0,00 a −0,06.

**Qualidade absoluta do estático com τ = 0,1** (média dos 15 textos):

| tradutor | saída cedo | ΔNLL |
|---|---|---|
| estreito | 21% | +0,25 |
| amplo | 22% | **+0,09** |

## Achados

1. **A crítica estava em boa parte certa.** O tradutor amplo, treinado offline, já resolve sozinho de −0,8 a −1,2 nats fora do domínio. É quase todo o ganho que o MYELIN tinha com o tradutor estreito. O que mais encolheu foi o viés: de −0,6/−0,8 para perto de zero, porque o erro global de idioma deixou de existir.
2. **Mas um ganho sobrevive, e é o do contexto.**
   - Com o tradutor amplo, o kNN ainda tira −0,13 a −0,20 nats em todos os grupos que não são prosa inglesa.
   - **Em Python, ele não mudou nada** (−0,21 com o estreito, −0,20 com o amplo). Python já estava no treino dos dois tradutores, então lá o ganho sempre foi só memória do contexto.
   - Com um tradutor bom, essa é a contribuição real da memória rápida: ~−0,15 nats na mesma taxa de saída, onde o texto tem estrutura que se repete (código, documentos técnicos, texto em outro idioma).
3. **Adaptar durante a leitura ≈ treinar antes.** O MYELIN com o tradutor estreito (comparação 2) chega a +0,04/+0,06 do tradutor amplo nos três grupos fora do domínio.
   - Ou seja, lendo 2048 tokens, sem nenhum dado do domínio antes, ele recupera quase tudo o que o retreino offline com ~1,5 milhão de tokens daqueles domínios dá.
   - É o argumento para quando não se sabe o domínio com antecedência.
4. **O MYELIN continua ≥ cache neural.** Com o tradutor amplo:

   | grupo | kNN 256 | cache neural |
   |---|---|---|
   | pt | −0,14 | −0,05 |
   | idiomas | −0,14 | −0,04 |
   | Python | −0,20 | −0,07 |
   | código-outro | −0,13 | −0,11 (empate) |
   | prosa-en | −0,03 | −0,03 (empate) |

5. **Os dois juntos são o melhor:** tradutor amplo + kNN dá −1,31 nats nos idiomas em relação ao ponto de partida. As duas coisas se somam, em vez de uma substituir a outra.
6. **Com o tradutor amplo, o viés lento deve sair.** Ele quase não ajuda e piora a prosa inglesa (+0,04). A configuração recomendada passa a ser **kNN 256 + auditoria**.

**Ressalvas:**
- 3 textos por grupo e uma semente.
- O tradutor amplo piorou um pouco a prosa inglesa (+0,07): a capacidade do $P_{17}$ ficou dividida.
- O cache neural continua sem ajuste de λ e β.

## O que isso significa

- **O resultado ficou menor e mais honesto.** O "quase o dobro de saídas" do Experimento 4 era, em grande parte, um tradutor estreito.
- O que o MYELIN de fato faz, com um tradutor bom, é **adaptação ao contexto**: ~−0,15 nats em código e em textos estruturados, de forma consistente e acima do cache neural.
- Quando o tradutor é estreito ou o domínio é desconhecido, ele recupera quase todo o efeito de um retreino offline.

## Próximo: Experimento 6, rascunho para decodificação especulativa

Com o tradutor amplo como base (a linha de base mais forte), medir a **taxa de aceitação** de rascunhos da camada rasa, com o modelo completo verificando todo token:
- a qualidade fica idêntica à do modelo completo;
- o professor está disponível em todo token;
- comparar estático, kNN 256 e cache neural, em profundidades mais rasas que k = 17, com o speedup estimado.
