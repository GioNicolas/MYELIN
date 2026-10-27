# MYELIN: memória rápida em tempo de teste para camadas rasas de LLMs congelados

**M.Y.E.L.I.N.** (*Memory-Yielded Early-Layer Inference Network*)  
**Autor:** Giorgio Nicolas de A. Bertino  
**Início:** 19 de outubro de 2026  
**Status:** pesquisa em andamento. Experimentos 0 a 7b concluídos; o próximo ciclo de hipóteses está em [`docs/fronteira-e-hipoteses.md`](docs/fronteira-e-hipoteses.md).

---

## A pergunta

Um LLM gasta todas as camadas em todo token, mesmo quando o mesmo padrão já apareceu muitas vezes na sessão. Os pesos ficam congelados durante o uso, então nada do que a camada final "descobre" chega às camadas rasas.

O MYELIN testa uma ideia inspirada na mielinização: **uma memória rápida numa camada rasa $k$ de um modelo congelado, ensinada durante a leitura pela camada final, sem gradiente.**
- Quando a camada rasa está em dúvida, o modelo completo responde.
- A diferença entre as duas respostas vira uma correção guardada na memória.
- Na próxima vez que uma situação parecida aparecer, a camada rasa já sai corrigida.

O alvo é transformar essa concordância extra em **velocidade**: saída cedo e, depois, rascunho de decodificação especulativa, em que a saída fica idêntica à do modelo completo.

A formulação matemática (sinapse rápida, portão de estresse, tradutor $P_k$, memória kNN de correções) está em [`docs/especificacao-matematica-e-experimento.md`](docs/especificacao-matematica-e-experimento.md).

## O que os experimentos mostraram até aqui

Todos em Qwen2.5-0.5B e 1.5B congelados, com protocolo de validação e teste separados, controles e revisão externa do código.

| | resultado |
|---|---|
| ✅ | A memória melhora de forma consistente a concordância da camada rasa com o modelo completo (Experimentos 2 a 7). No rascunho especulativo, a aceitação sobe de 21,5% para 37,6% na camada 8. |
| ✅ | O ganho cresce com a distância até o domínio de treino do tradutor. Adaptar durante a leitura recupera quase tudo o que um retreino offline dá (Experimento 4b). |
| ✅ | O estado oculto já calculado escolhe bem **de onde copiar** o contexto (o "reflexo"). Na frente da busca de n-gramas, ganha em 20 de 24 respostas de chat (Experimento 7b). É uma reinvenção independente do PLD+ (arXiv 2412.01447). |
| ❌ | No 0,5B, com o custo contado corretamente, a cabeça rasa com memória **não acelera** em texto realista (no máximo 0,90×): a cabeça custa 27% de um passo. |
| ❌ | A camada que concorda ~50% com a final fica a ~70% da profundidade em qualquer tamanho testado (Experimento 5). |
| ⚠️ | Em chat realista, quase todo o ganho da memória é uma correção global (Experimento 7b, controles). |

## Documentos

- 📐 **[Especificação matemática](docs/especificacao-matematica-e-experimento.md):** as equações da sinapse rápida ($\Delta W_{\text{fast}}$), do portão de estresse ($\mathcal{S}_t$), do aprendizado pelo resíduo entre a camada final e a rasa, e do decaimento.
- 🧭 **[Fronteira e hipóteses](docs/fronteira-e-hipoteses.md):** o mapa dos trabalhos próximos (PLD+, AdaPLD, SwitchSD, adaptação online de rascunhadores), o que só este projeto mediu e o portfólio de hipóteses seguintes, cada uma com o seu critério de morte.
- 🔬 **[Experimento 0](docs/resultados-exp0.md):** as camadas rasas lidas direto pela cabeça final concordam 0% com a camada final, o que torna obrigatório o tradutor $P_k$ (*tuned lens*). Com ele, a camada 17 de 24 já concorda em 39–61%.
- 🧪 **[Experimento 1](docs/resultados-exp1.md):** a primeira curva de mielinização. A sinapse ajuda no texto em português, mas colapsa no código.
- 🧭 **[Experimento 2](docs/resultados-exp2.md):** com o modelo completo num calendário fixo, uma memória exata (kNN) melhora a camada rasa em 7 de 7 textos, mas pouco: fecha 7–19% da distância. O colapso vinha do portão, não das chaves.
- ⚖️ **[Experimento 3](docs/resultados-exp3.md):** com o portão de volta, a memória dá 6 a 13 pontos a mais de saídas com a mesma qualidade. No 0,5B, porém, nenhum ponto de operação fica mais barato que o modelo completo.
- 🌍 **[Experimento 4](docs/resultados-exp4.md):** em 15 textos, o ganho segue a distância até o treino do tradutor: quase o dobro de saídas cedo fora do domínio, zero em inglês.
- ⚗️ **[Experimento 4b](docs/resultados-exp4b.md):** com um tradutor amplo, a maior parte do ganho fora do domínio some (era compensação do tradutor estreito). O que sobra é memória do contexto, acima do cache neural.
- 📏 **[Experimento 5](docs/resultados-exp5.md):** no Qwen2.5-1.5B a cabeça pesa metade, mas a saída cedo num modelo congelado continua sem pagar a conta.
- 🚀 **[Experimento 6](docs/resultados-exp6.md):** a camada rasa como rascunho especulativo. A memória sobe a aceitação em 15 de 15 textos. **Correção após revisão externa:** o modelo de custo omitia um passo, e com a conta certa nada acelera no 0,5B.
- 🔁 **[Experimento 7](docs/resultados-exp7.md):** o ciclo real da especulação, com custo corrigido e linhas de base justas.
  - Com a memória, a camada 8 passa a acelerar (1,21×, 15/15 textos), e três controles confirmam o mecanismo.
  - O reflexo ganha da busca de n-gramas: 4,62× contra 4,29×.
  - Ressalva: o texto gerado é 77%–100% laço.
- 💬 **[Experimento 7b](docs/resultados-exp7b.md):** em 24 respostas de chat, o reflexo sozinho perde da busca de n-gramas (1,68× contra 1,73×), mas na frente dela ganha em 20 de 24. A cabeça rasa com memória não acelera.

## Estrutura

```text
src/myelin/      sinapse rápida, portão de estresse, tradutor P_k, memória kNN, cache neural,
                 controles e o simulador exato do ciclo de decodificação especulativa
experiments/     um script por experimento (validação escolhe hiperparâmetros, teste mede)
notebooks/       os mesmos experimentos para rodar em GPU no Colab/Kaggle
data/            textos de avaliação congelados (fontes e licenças em data/eval/FONTES.md e data/eval_dominio/FONTES.md)
tests/           testes em CPU com modelos de brinquedo
docs/            especificação, resultados de cada experimento e hipóteses
```

## Como rodar

```bash
pip install -e ".[dev]"
pytest -q                 # testes em CPU, modelos de brinquedo
```

Os experimentos com LLMs reais precisam de GPU: abra o notebook correspondente em `notebooks/` no Colab ou no Kaggle (T4).
