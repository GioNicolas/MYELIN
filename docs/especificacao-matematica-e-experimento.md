# Especificação Matemática, Hipóteses e Plano Experimental — MYELIN

**Projeto:** **MYELIN** (*Memory-Yielded Early-Layer Inference Network*)  
**Autor:** Giorgio Nicolas de A. Bertino  
**Data:** 19 de outubro de 2026

---

## 1. Formulação Matemática das 3 Peças do Algoritmo

Considere uma rede Transformer pré-treinada com $L$ camadas (ex.: $L = 28$ ou $32$) cujos pesos originais estão **100% congelados**. Escolhemos uma camada rasa $k \ll L$ (ex.: $k = 4$ ou $k = 6$) como o nosso **Nó de Reflexo (Cerebelo)**.

### Peça 1: A Mochila de Sinapses Rápidas ($\Delta W_k$)
Na camada rasa $k$, acoplamos em paralelo ao bloco original congelado $f_k(\mathbf{h})$ um adaptador sináptico ultraleve de posto baixo (*Fast-Weight Low-Rank Synapse*, com dimensão interna $r \ll d$, ex.: $r = 16$):

$$\hat{\mathbf{h}}_t^{(k)} = f_k\left(\mathbf{h}_t^{(k-1)}\right) + \underbrace{\mathbf{B}_k \left(\mathbf{A}_k^\top \mathbf{h}_t^{(k-1)}\right)}_{\text{Sinapse Rápida } \Delta W_k(\mathbf{h})}$$

* No início de uma sessão ($t = 0$), $\mathbf{B}_k = \mathbf{0}$, de modo que a rede se comporta exatamente como o modelo original de fábrica.
* O tamanho de $(\mathbf{A}_k, \mathbf{B}_k)$ é fixo e minúsculo (menos de $0.5\%$ dos parâmetros de uma camada, ~1 a 2 MB). **Nunca cresce com o tempo.**

---

### Peça 2: O Medidor de "Estresse" ($\mathcal{S}_t$) e Portão de Decisão
A partir do estado da camada rasa $\hat{\mathbf{h}}_t^{(k)}$, aplicamos a cabeça de leitura (*LM Head* compartilhada do próprio modelo ou um projetor leve $P_k$) para obter a distribuição rápida de probabilidades sobre a próxima palavra:

$$p_t^{(k)} = \operatorname{Softmax}\left(\mathbf{W}_{\text{head}} \cdot \operatorname{Norm}\left(\hat{\mathbf{h}}_t^{(k)}\right)\right)$$

Definimos o **Índice de Estresse Sináptico** $\mathcal{S}_t \in [0, 1]$ como a entropia normalizada (ou margem de incerteza entre os 2 maiores candidatos):

$$\mathcal{S}_t = -\frac{1}{\log |V|} \sum_{v \in V} p_t^{(k)}(v) \log p_t^{(k)}(v)$$

* **Regra de Decisão (O Portão):**
  * **Se $\mathcal{S}_t \le \tau_{\text{stress}}$ (Baixo Estresse / Reflexo Consolidado):**  
    O modelo aceita imediatamente a previsão da camada rasa $p_t^{(k)}$ (ou pula os blocos pesados das camadas $k+1 \dots L$). **Economia computacional máxima!**
  * **Se $\mathcal{S}_t > \tau_{\text{stress}}$ (Alto Estresse / Dúvida):**  
    O sinal sobe pelo **Córtex Profundo** (camadas $k+1$ até $L$), produzindo o estado final verdadeiro $\mathbf{h}_t^{(L)}$ e a distribuição de elite $p_t^{(L)}$.

---

### Peça 3: Mielinização — Aprendizado Instantâneo pelo Resíduo ($L \rightarrow k$)
Sempre que o estresse foi alto ($\mathcal{S}_t > \tau_{\text{stress}}$) e a Camada Profunda $L$ precisou trabalhar, nós não desperdiçamos esse esforço! Usamos a resposta da Camada $L$ como gabarito instantâneo para atualizar a Sinapse Rápida $(\mathbf{A}_k, \mathbf{B}_k)$ lá na Camada $k$:

1. **Cálculo do Resíduo Interno (No Espaço de Estados ou Logits):**
   $$\mathbf{r}_t = \operatorname{Norm}\left(\mathbf{h}_t^{(L)}\right) - \operatorname{Norm}\left(\hat{\mathbf{h}}_t^{(k)}\right) \quad \text{ou} \quad \mathcal{L}_{\text{distill}} = D_{\text{KL}}\left(p_t^{(L)} \parallel p_t^{(k)}\right)$$
2. **Regra de Atualização Sináptica Local (Sem Backprop pelas 32 Camadas!):**
   Com um custo de apenas uma multiplicação de vetores (fração de milissegundo), aplicamos a atualização com **Homeostase (Decaimento Elástico $\gamma \in [0.98, 0.999]$)**:
   $$\mathbf{B}_k \leftarrow \gamma \mathbf{B}_k + \eta \cdot \mathbf{r}_t \left(\mathbf{A}_k^\top \mathbf{h}_t^{(k-1)}\right)^\top$$
   *(Repare na elegância dessa equação: é uma regra de aprendizado **Hebbiana modulada por erro / Delta Rule** pura! Não precisa nem chamar o autograd pesado do PyTorch se usarmos o resíduo direto $\mathbf{r}_t$!)*

---

## 2. As 3 Hipóteses Científicas que Vamos Filtrar e Testar no Código

1. **Hipótese H1 (Eficiência Progressiva — "A Curva da Mielina"):**  
   À medida que o modelo processa um documento técnico, repositório de código ou sessão de resolução de problemas, a taxa de acionamento das camadas profundas cai de $>80\%$ nos primeiros parágrafos para $<35\%$ nos parágrafos seguintes, reduzindo os FLOPs totais em mais de $2\times$.
2. **Hipótese H2 (Superioridade sobre Early-Exit Estático):**  
   Com o mesmo orçamento médio de camadas (ex.: média de 10 camadas por token), o **MYELIN** (com sinapse rápida viva $\Delta W_k$) atinge perplexidade significativamente menor e acurácia maior do que o *Early-Exit Estático* ($\Delta W_k = 0$), porque aprende os padrões específicos do contexto atual.
3. **Hipótese H3 (Estabilidade Homeostática):**  
   O fator de elasticidade $\gamma < 1$ impede que a sinapse rápida sature quando o assunto muda no meio da conversa, permitindo adaptação contínua sem esquecimento catastrófico.

---

## 3. Estrutura do Repositório para Iniciar a Implementação

```text
MYELIN/
├── README.md                                          # Visão geral e posicionamento
├── docs/
│   ├── guia-de-estudo-e-pitch.md                      # Guia para estudo e entrevistas
│   └── especificacao-matematica-e-experimento.md      # Este documento
├── src/
│   └── myelin/
│       ├── __init__.py
│       ├── fast_synapse.py     # Camada FastSynapse (Regra Delta/Hebbiana rápida sobre o resíduo r_t)
│       ├── stress_gate.py      # Cálculo de Estresse (Entropia / Margem) e roteamento Camada k vs L
│       └── wrapper.py          # Acoplador para modelos HuggingFace (Qwen / Llama)
├── notebooks/
│   └── experimento_curva_mielina.ipynb                # Notebook pronto para gerar o gráfico decisivo
└── tests/
    └── test_fast_synapse.py                           # Teste matemático local em CPU (2 segundos)
```
