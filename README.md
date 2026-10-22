# MYELIN: Consolidação Sináptica Guiada por Estresse para Inferência em Camadas Rasas

**Sigla Científica:** **M.Y.E.L.I.N.** (*Memory-Yielded Early-Layer Inference Network*)  
**Subtítulo do Artigo:** *Stress-Driven Deep-to-Shallow Synaptic Consolidation at Test Time*  
**Autor:** Giorgio Nicolas de A. Bertino  
**Data de início:** 19 de outubro de 2026  
**Status:** Projeto de pesquisa fundamental em eficiência e plasticidade sináptica para redes neurais e LLMs.

---

## Documentos deste Repositório

- 📘 **[Guia de Estudo, Explicação Visual e Pitch (Caso Alguém Pergunte)](docs/guia-de-estudo-e-pitch.md):** documento feito para estudar a fundo como o MYELIN funciona sem jargão difícil, entender o que é uma sinapse no código, por que a memória nunca explode e como responder perguntas de avaliadores, professores ou recrutadores.
- 📐 **[Especificação Matemática, Hipóteses e Experimento Decisivo](docs/especificacao-matematica-e-experimento.md):** equações da sinapse rápida ($\Delta W_{\text{fast}}$), gatilho de estresse ($\mathcal{S}_t$), aprendizado pelo resíduo entre a Camada 32 e a Camada 4, decaimento homeostático e arquitetura de código (`src/myelin/`).
- 🔬 **[Resultados do Experimento 0](docs/resultados-exp0.md):** no Qwen2.5-0.5B real, as camadas rasas lidas direto pela cabeça final concordam 0% com a camada final e ficam confiantes e erradas, o que torna obrigatório o tradutor $P_k$ (*tuned lens*). Com ele, a camada 17 de 24 já concorda em 39–61%, mas o portão de estresse ainda precisa de calibração.
- 🧪 **[Resultados do Experimento 1](docs/resultados-exp1.md):** a primeira Curva de Mielinização no Qwen real. A sinapse ajuda no texto em português, mas colapsa no código quando as chaves ficam quase paralelas; a correção está em teste.
- 🧭 **[Resultados do Experimento 2](docs/resultados-exp2.md):** com o córtex num calendário fixo, uma memória exata (kNN) melhora a camada rasa em 7 de 7 textos, mas pouco (fecha 7–19% da distância até o modelo completo); a memória linear com chaves descorrelacionadas atrapalha; o colapso vinha do portão, não das chaves.
- ⚖️ **[Resultados do Experimento 3](docs/resultados-exp3.md):** com o portão de volta, a memória melhora a curva custo × qualidade (6 a 13 pontos a mais de saídas com a mesma qualidade), mas o ganho vem do português e do código repetitivo do projeto, quase zero em inglês. No 0,5B nenhum ponto de operação fica mais barato que o modelo completo, porque a cabeça custa ~9 camadas.
- 🌍 **[Resultados do Experimento 4](docs/resultados-exp4.md):** em 15 textos, o ganho segue a distância até o treino do tradutor: fora do domínio (idiomas, português, código não-Python), quase o dobro de saídas cedo com a mesma qualidade, ganhando do cache neural; em inglês, zero. Nos idiomas o ganho é sobretudo correção global; no código, memória do contexto.
- 📏 **[Resultados do Experimento 5](docs/resultados-exp5.md):** no Qwen2.5-1.5B a cabeça pesa metade (5 de 28 camadas), mas o early exit num modelo congelado continua sem pagar a conta: a camada que concorda ~50% com a final fica a ~70% da profundidade em qualquer tamanho. O ganho da memória fora do domínio se repete (português −1,0 nat).
- ⚗️ **[Resultados do Experimento 4b](docs/resultados-exp4b.md):** com um tradutor amplo (vários idiomas e linguagens), a maior parte do ganho fora do domínio some: era compensação do tradutor estreito. O que sobra é memória do contexto, −0,13 a −0,20 nats em código e textos fora do inglês, acima do cache neural. E adaptar durante a leitura recupera quase tudo o que o retreino offline do tradutor dá.
- 🚀 **[Resultados do Experimento 6](docs/resultados-exp6.md):** com a camada rasa como rascunho de decodificação especulativa (a saída fica idêntica à do modelo completo), a memória rápida, ensinada pelo verificador, sobe a aceitação do rascunho em 15 de 15 textos: de 21,5% para 37,6% na camada 8, e até o dobro em código. Ganha com folga do cache neural. Speedup estimado ainda modesto no 0,5B (até 1,37× em código; 1,62× com a cabeça de um 7B).

---

## 1. A Inspiração Biológica e o Problema das LLMs Atuais

### O Absurdo Computacional das LLMs de Hoje
Quando uma LLM moderna (como GPT-4, Llama-3 ou Qwen) lê ou escreve um texto, ela sofre de dois defeitos estruturais:
1. **Esforço Máximo Uniforme:** Ela aciona **100% das suas 32 a 80 camadas profundas** para cada palavra, gastando a mesma energia absurda tanto para prever uma vírgula óbvia quanto para resolver um teorema complexo.
2. **Sinapses Congeladas ("Mortas") na Conversa:** Mesmo que a IA resolva o mesmo problema difícil 50 vezes seguidas durante um projeto, as camadas iniciais dela não aprendem nada em tempo real. Nas 50 vezes, ela será obrigada a subir até a última camada.

### Como o Cérebro de uma Criança Faz Diferente (A Mielinização)
No cérebro humano, existe uma divisão entre o **Córtex** (camadas profundas, lentas e conscientes) e o **Cerebelo / Subcórtex** (camadas rasas de reflexo rápido):
* Na primeira vez que você tenta aprender algo novo, as camadas rasas sentem **"Estresse" (alta dúvida/erro)** e acordam as camadas profundas do Córtex.
* Quando o Córtex descobre a resposta certa, ele envia um sinal de erro (resíduo) que modifica as **sinapses rápidas das camadas rasas**, envolvendo aquele circuito em **Mielina (*Myelin*)**.
* **Resultado:** Nas próximas vezes em que aquele padrão aparece, as primeiras camadas já sabem o atalho! As camadas profundas nem precisam ligar, reduzindo o gasto de energia em até 10x sem criar um "banco de dados infinito".

---

## 2. Como Funciona o Algoritmo MYELIN (Em 3 Passos)

```text
[Palavra / Entrada x_t]
          │
          ▼
┌───────────────────────────────────────────────────────────────────┐
│ 1. CAMADA RASA (Ex.: Camada 4 + Sinapse Rápida Móvel ΔW_fast)     │
│    Calcula previsão rápida ŷ_4 e mede o "Estresse" S_t (Dúvida)   │
└─────────┬───────────────────────────────────────────┬─────────────┘
          │                                           │
   [Se Estresse Baixo]                         [Se Estresse Alto]
   (Já sabe o atalho!)                         (Caso novo / difícil!)
          │                                           │
          ▼                                           ▼
┌─────────────────────────┐         ┌───────────────────────────────┐
│ SAÍDA IMEDIATA!         │         │ 2. ACORDA O CÓRTEX PROFUNDO   │
│ Economiza as outras     │         │    Processa até a Camada 32   │
│ 28 camadas da rede!     │         │    e obtém a resposta y_32    │
└─────────────────────────┘         └─────────────────┬─────────────┘
                                                      │
                                                      ▼
                                    ┌───────────────────────────────┐
                                    │ 3. MIELINIZAÇÃO (Resíduo)     │
                                    │    Calcula Erro = y_32 - ŷ_4  │
                                    │    e gira os botões da        │
                                    │    Sinapse Rápida ΔW_fast     │
                                    │    lá na Camada 4 na hora!    │
                                    └───────────────────────────────┘
```

---

## 3. Onde Está a Contribuição Científica Inédita (Literatura 2024–2026)

*Verificado no arXiv em 19/10/2026: o nome **MYELIN** e a combinação de **Early-Exit Dinâmico + Destilação Sináptica em Tempo de Execução das Camadas Profundas para as Rasas** são inéditos.*

Hoje existem duas linhas separadas na ciência que nunca foram unidas:

| Linha de Pesquisa Existente | Artigos de Referência (2024–2026) | O que fazem | Onde falham (E o que o MYELIN resolve) |
| :--- | :--- | :--- | :--- |
| **Ilha 1: Early-Exit Estático** (Sair nas camadas rasas) | *LayerSkip* (Meta, 2024); *Early Exit Is a Natural Capability* ([arXiv:2412.01455](https://arxiv.org/abs/2412.01455)); *ADEPT* (jan/2026, [arXiv:2601.03700](https://arxiv.org/abs/2601.03700)) | Palavras fáceis saem na Camada 4; palavras difíceis vão até a Camada 32. | **As sinapses da Camada 4 são congeladas!** Se o mesmo conceito difícil aparecer 50 vezes na conversa, ele falha na Camada 4 as 50 vezes e nunca aprende o atalho. |
| **Ilha 2: Test-Time Training / Sinapses Rápidas** | *Titans* (Google, 2025, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)); *In-Place TTT* (abr/2026) | Atualizam sinapses rápidas durante a conversa guiadas pela surpresa. | **Rodam todas as 32 camadas o tempo todo!** Usam a sinapse rápida apenas como memória de texto longo, nunca para ensinar as camadas rasas a desligarem as camadas profundas. |
| **Nossa União: MYELIN** | **Este projeto (2026)** | A Camada 32 treina a Sinapse Rápida da Camada 4 em tempo real quando há estresse. | **Quanto mais a IA trabalha num domínio/conversa, menos camadas ela precisa usar!** O custo computacional despenca progressivamente ao longo do uso. |

---

## 4. POR ONDE COMEÇAR (Roteiro Prático para o Próximo Chat)

### Recursos Necessários
* **Sem supercomputador:** O algoritmo se acopla a qualquer modelo aberto pequeno (ex.: `Qwen-2.5-0.5B` / `1.5B` ou `Llama-3.2-1B`, que têm entre 16 e 28 camadas).
* **Desenvolvimento local:** Python 3.12 na sua máquina para criar a camada de Sinapse Rápida (`FastSynapseLoRA`) e os testes unitários em CPU.
* **Bateria de avaliação rápida:** Exportação de Notebook `.ipynb` para rodar em poucos minutos na GPU gratuita do Google Colab / Kaggle.

### O Primeiro Experimento ("A Curva de Mielinização")
Dar para o modelo um texto técnico ou código com padrões recorrentes e medir, do Parágrafo 1 ao Parágrafo 20:
1. **Profundidade Média Utilizada (Nº de camadas acesas por token):** provando que no Parágrafo 1 o modelo usa ~100% das camadas e no Parágrafo 10 já resolve a maioria dos tokens na Camada Rasa.
2. **Qualidade / Perplexidade / Acurácia:** provando que o modelo com **MYELIN** mantém a inteligência da Camada 32 gastando uma fração dos cálculos, superando o *Early-Exit Estático* (sem sinapse viva).

### Comando para Iniciar no Novo Chat:
> *"Leia o `README.md`, o `docs/guia-de-estudo-e-pitch.md` e o `docs/especificacao-matematica-e-experimento.md`. Vamos começar a Fase 1 do projeto MYELIN: me ajude a estudar ou crie o módulo inicial em `src/myelin/` com a Sinapse Rápida na camada rasa aprendendo pelo resíduo da camada profunda."*
