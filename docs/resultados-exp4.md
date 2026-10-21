# Resultados — Experimento 4: a memória rápida como adaptação ao domínio

**Data:** 21 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B`, pesos congelados, GPU T4 do Kaggle (~36 min)  
**Configuração:**
- k = 17 com o tradutor $P_{17}$, portão de estresse com τ de 0,1 a 0,5;
- auditoria de 5% nas memórias que aprendem com o córtex;
- 15 textos de 2048 tokens em [`data/eval_dominio/`](../data/eval_dominio/FONTES.md), 3 por grupo.

**Grupos:**
- fora do treino do tradutor: português, outros idiomas (es, de, fr), código não-Python (Rust, Go, JS);
- controles, do mesmo tipo do treino: Python e prosa em inglês (WikiText, split de teste).

**Configurações:**
- estático;
- viés lento (η = 0,03);
- kNN 256;
- viés lento + kNN 256;
- cache neural 256 (Grave et al., 2017; λ = 0,1, β = 20, sem ajuste). O cache aprende o próximo token real de todo token e não usa o córtex.

**Scripts:** [`experiments/exp4_dominio.py`](../experiments/exp4_dominio.py), [`experiments/exp3_analise.py`](../experiments/exp3_analise.py) · **Notebook:** [`notebooks/exp4_dominio.ipynb`](../notebooks/exp4_dominio.ipynb)

---

## Ganho na mesma taxa de saída, por grupo

ΔNLL(config) − ΔNLL(estático interpolado na mesma taxa de saída), em nats. Negativo = melhor.

| grupo | viés lento | kNN 256 | **viés + kNN 256** | cache neural |
|---|---|---|---|---|
| outros idiomas | −0,84 | −0,92 | **−1,12** | −0,39 |
| português | −0,64 | −0,58 | **−0,86** | −0,25 |
| código não-Python | −0,26 | −0,78 | **−0,83** | −0,45 |
| Python (controle) | −0,04 | −0,21 | **−0,23** | −0,08 |
| prosa em inglês (controle) | +0,05 | +0,00 | +0,05 | **−0,02** |

**Por texto (viés + kNN):**
- todos os 9 textos fora do domínio melhoram: −0,56 (`option.rs`) a −1,29 (Kafka);
- os 3 de Python melhoram pouco: −0,13 a −0,34;
- os 3 de prosa em inglês pioram um pouco: +0,04 a +0,08.

**Com a mesma qualidade do estático, fora do domínio, quase o dobro de saídas cedo** (interpolando a curva do viés + kNN):

| grupo | ΔNLL fixo | estático | viés + kNN |
|---|---|---|---|
| português | +1,37 | 30% | ~53% |
| outros idiomas | +1,73 | 26% | ~53% |
| código não-Python | +0,99 | 52% | ~81% |

A precisão das saídas cedo (média dos 15 textos, τ = 0,3) sobe de 51% (estático) para 62% (viés + kNN).

## Resumo (média dos 15 textos)

| config | τ = 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|
| estático | 21% / +0,25 | 35% / +0,60 | 48% / +1,06 | 63% / +1,72 | 77% / +2,44 |
| viés lento | 26% / +0,26 | 43% / +0,64 | 61% / +1,14 | 76% / +1,76 | 87% / +2,19 |
| kNN 256 | 31% / +0,23 | 44% / +0,48 | 57% / +0,87 | 72% / +1,42 | 85% / +2,01 |
| viés + kNN 256 | 31% / +0,21 | 45% / +0,46 | 59% / +0,82 | 74% / +1,32 | 87% / +1,77 |
| cache neural | 19% / +0,12 | 33% / +0,39 | 48% / +0,80 | 63% / +1,39 | 79% / +2,09 |

Cada célula é *saída cedo / ΔNLL*. O custo real é o mesmo do Experimento 3, porque depende só da taxa de saída: abaixo de 100% apenas com ΔNLL acima de ~1 nat.

## Achados

1. **A hipótese da adaptação ao domínio se confirma, com 3 textos por grupo.** O ganho segue exatamente a distância até o treino do tradutor:
   - idiomas e português: −0,9 a −1,1 nats;
   - código não-Python: −0,8;
   - Python: −0,2;
   - prosa em inglês: zero.

   Fora do domínio, todos os 9 textos melhoram.
2. **Dois mecanismos diferentes, e os dois importam.**
   - **Nos idiomas, a maior parte é correção global.** O viés lento sozinho já tira −0,6 a −0,8, porque o tradutor erra o idioma inteiro na mesma direção.
   - **No código, a maior parte é memória do contexto.** O viés quase não ajuda (−0,26) e o kNN tira −0,78, porque identificadores e padrões se repetem dentro do arquivo.
   - **A combinação é a melhor em todo grupo fora do domínio.**
3. **O MYELIN ganha do cache neural fora do domínio**, por 2 a 3 vezes: −0,86 contra −0,25 em português, −1,12 contra −0,39 nos idiomas, −0,83 contra −0,45 no código. O cache usa mais informação (o token real de todo token) e mesmo assim perde. Só empata ou ganha, por pouco, em prosa inglesa, onde nada ajuda.
   - Uma diferença de comportamento: o cache baixa a NLL, mas **não aumenta a taxa de saída** (48% → 48% com τ = 0,3), porque misturar λ = 0,1 de outra distribuição não deixa a previsão mais confiante. A correção no estado oculto, a do MYELIN, deixa.
4. **A adaptação é rápida e não acumula ao longo do texto.** Por quarto (512 tokens cada), a vantagem aparece já no primeiro quarto e fica estável. Com memória de 256 consultas, o que dá para aprender é aprendido logo.
5. **No domínio do tradutor, o viés atrapalha um pouco** (+0,05 em prosa inglesa) e não deve ficar ligado sempre. Uma saída possível é ligar o viés só quando o resíduo médio do córtex for grande.

**Ressalvas:**
- O cache neural rodou numa configuração só (λ = 0,1, β = 20). Grave et al. ajustam λ e θ num conjunto de validação, então um cache ajustado pode fechar parte da diferença.
- **O confundidor principal:** boa parte do ganho fora do domínio compensa um tradutor treinado só em inglês e Python. Um tradutor treinado offline em dados multilíngues e multi-linguagem pode tirar essa vantagem, principalmente a do viés. A parte do kNN no código, que é memória do contexto, deve sobreviver melhor.
- A interpolação linear sobre uma curva convexa exagera um pouco os ganhos pequenos (ver Experimento 3).

## O que isso significa

- **O MYELIN funciona como adaptação ao domínio em tempo de teste.** Com o modelo congelado e memória fixa (~1,8 MB), a leitura rasa se ajusta a um idioma ou a uma linguagem de programação que o tradutor nunca viu. Fora do domínio, isso dá quase o dobro de saídas cedo com a mesma qualidade, e ganha do cache neural.
- **A economia de custo continua sendo do Experimento 5.** No 0,5B, a cabeça ainda torna todo ponto de qualidade razoável mais caro que o modelo completo.

## Próximos passos

1. **Experimento 5 (Qwen2.5-1.5B):** a conta de custo fecha num modelo maior?
2. **Tradutor multilíngue:** retreinar o $P_{17}$ com prosa em vários idiomas e código em várias linguagens, e rodar o Experimento 4 de novo. É o teste que separa "o MYELIN adapta ao domínio" de "o MYELIN compensa um tradutor estreito".
3. **Cache neural ajustado** (alguns λ e β), para uma comparação justa com a literatura.
