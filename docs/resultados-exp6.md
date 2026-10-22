# Resultados — Experimento 6: a camada rasa como rascunho de decodificação especulativa

> ## ⚠️ Correção (revisão externa, 22/10/2026)
>
> A revisão por dois modelos externos (Codex e Antigravity) encontrou um **erro no modelo de custo**. As seções de speedup abaixo estão **erradas** e ficam só como registro.
>
> **O erro:** a fórmula cobrava a verificação como (L−k+H)/(L+H). Mas o último token rascunhado ainda precisa passar pelas camadas 1..k antes de as γ+1 posições passarem juntas pelas camadas de cima (o LayerSkip faz isso em `forward_remainder`). Então a verificação custa **um passo inteiro**: custo do ciclo = γ·(k+H)/(L+H) + 1. Com γ = 1, a conta errada dava custo independente de k.
>
> **Speedup guloso corrigido** (estático → kNN 1024; com a cabeça real | com a cabeça de 7B):
>
> | | todos | código não-Python | Python |
> |---|---|---|---|
> | k = 8 | 0,80× → 0,91× \| 0,88× → 0,99× | 0,83× → 1,00× \| 0,91× → 1,10× | 0,80× → 0,96× \| 0,87× → 1,05× |
> | k = 17 | 0,82× → 0,87× \| 0,84× → 0,90× | 0,91× → 0,97× \| 0,94× → 1,01× | 0,89× → 0,96× \| 0,92× → 0,99× |
>
> - **Nenhuma configuração acelera no 0,5B.** Com γ = 1, empatar exige aceitação maior que o custo do rascunho: 52% em k = 8 e 79% em k = 17, com a cabeça real.
> - **O que continua valendo:** o ganho de **concordância** do rascunho com o modelo completo (15/15 textos, 21,5% → 37,6% em k = 8), medido em teacher forcing.
>
> **Outros pontos aceitos da revisão:**
> - A concordância média sobre todas as posições não é a aceitação por ciclo. Num decodificador real, as posições logo após um acerto são puladas (o token bônus). Falta uma simulação do ciclo, em texto gerado pelo próprio modelo.
> - O cache neural foi uma linha de base injusta: aprendeu o token do texto, e não o do verificador; λ = 0,1 quase nunca consegue mudar o top-1; não tinha porta de familiaridade; capacidade 256 contra 1024. "Corrigir o estado oculto é o que faz a diferença" **não está demonstrado**.
> - Faltam a linha de base de **busca de n-gramas no contexto** (Prompt Lookup Decoding, Cacheback), que propõe tokens sem rodar camada nenhuma, e controles da memória (resíduo médio, último resíduo, pares embaralhados).
> - **Exageros de redação:**
>   - "sem treinar nada": o tradutor $P_k$ é treinado uma vez, offline;
>   - "saída idêntica": é uma propriedade do algoritmo de verificação, que este experimento não implementou.
> - **Novidade:** online speculative decoding (Liu et al., 2024) e DVI (Bhansali e Heck, 2025, arXiv:2510.05421) já usam o verificador para ensinar o rascunhador online, com gradiente. O diferencial do MYELIN é fazer isso **sem retropropagação**, com uma memória não paramétrica de tamanho fixo.
>
> O plano de correção e o próximo experimento estão em [resultados-exp7.md](resultados-exp7.md) (a fazer).

**Data:** 22 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B` (24 camadas, cabeça ≈ 9,1 camadas), pesos congelados, GPU T4 do Kaggle (~29 min)  
**Configuração:**
- tradutor amplo $P_k$ ([Experimento 4b](resultados-exp4b.md)), rascunho nas camadas k = 8, 12 e 17;
- os 15 textos de `data/eval_dominio/` (2048 tokens, 3 por grupo), em teacher forcing;
- o modelo completo verifica todo token, e a memória aprende com o verificador depois de cada rascunho ser medido.

**Configurações:**
- estático (só $P_k$);
- kNN 256 e kNN 1024 (memória de correções do estado oculto, ~1,8 e ~7 MB);
- cache neural 256 (Grave et al., 2017; λ = 0,1);
- kNN 256 + cache neural.

**Métricas:**
- aceitação gulosa (top-1 do rascunho == top-1 do modelo completo);
- aceitação na amostragem α = Σ min(p_k, p_L) (Leviathan et al., 2023);
- speedup esperado:
  - c_rascunho = (k+H)/(L+H);
  - c_verificação = (L−k+H)/(L+H), reaproveitando as camadas 1..k como no LayerSkip;
  - com a cabeça real e com a cabeça de um 7B (2,3 de 28 camadas).

A saída final é idêntica à do modelo completo em todos os casos.

**Script:** [`experiments/exp6_rascunho.py`](../experiments/exp6_rascunho.py) · [`src/myelin/speculative.py`](../src/myelin/speculative.py) · **Notebook:** [`notebooks/exp6_rascunho.ipynb`](../notebooks/exp6_rascunho.ipynb)

---

## Aceitação gulosa do rascunho (média dos 15 textos)

| config | k = 8 | k = 12 | k = 17 |
|---|---|---|---|
| estático | 21,5% | 24,0% | 46,1% |
| kNN 256 | 33,7% | 35,4% | 54,1% |
| **kNN 1024** | **37,6%** | **39,2%** | **55,9%** |
| cache neural 256 | 28,3% | 29,0% | 48,0% |
| kNN 256 + cache neural | 34,0% | 35,3% | 53,7% |

**Consistência por texto:**
- o kNN 256 sobe a aceitação em **15 de 15 textos**, nas três profundidades;
- o kNN 1024 supera o kNN 256 em 14, 14 e 15 de 15 textos (k = 8, 12, 17);
- ganho por texto, kNN 256 com k = 8: de +2,5 pontos (WikiText *Ise battleship*) a +28,1 (`response.js`).

## Por grupo: aceitação e speedup guloso (estático → kNN 1024)

Cada célula é *aceitação · speedup com a cabeça real · speedup com a cabeça de 7B*.

| grupo | k = 8 | k = 17 |
|---|---|---|
| código não-Python | 26% · 0,99× · 1,18× → **52% · 1,19× · 1,41×** | 63% · 1,28× · 1,51× → **74% · 1,37× · 1,62×** |
| Python | 21% · 0,95× · 1,12× → **45% · 1,14× · 1,35×** | 59% · 1,25× · 1,48× → **71% · 1,34× · 1,59×** |
| português | 12% · 0,88× · 1,04× → 26% · 0,99× · 1,17× | 30% · 1,02× · 1,20× → 40% · 1,10× · 1,30× |
| outros idiomas | 13% · 0,88× · 1,05× → 23% · 0,97× · 1,15× | 23% · 0,97× · 1,15× → 35% · 1,06× · 1,25× |
| prosa inglesa | 35% · 1,06× · 1,26× → 41% · 1,11× · 1,31× | 56% · 1,22× · 1,45× → 59% · 1,25× · 1,48× |

**Ganho de aceitação sobre o estático, 2ª metade do texto, kNN 1024** (pontos percentuais):

| | pt | idiomas | código-outro | Python | prosa-en |
|---|---|---|---|---|---|
| k = 8 | +15,4 | +11,7 | **+27,9** | **+25,1** | +8,4 |
| k = 17 | +11,5 | +12,8 | +14,4 | +9,1 | +5,0 |

**Na média, com speedup guloso:**
- k = 17: de 1,15× (estático) para 1,22× (kNN 1024) com a cabeça real, e de 1,36× para 1,45× com a cabeça de 7B;
- k = 8: o kNN transforma 0,95× (pior que não especular) em 1,08×.

O melhor rascunho foi sempre de γ = 1 token, porque a aceitação ainda é baixa para rascunhos longos.

## Achados

1. **A memória rápida aumenta muito a aceitação do rascunho, e em todo texto.**
   - Na camada 8, a aceitação sobe de 21,5% para 37,6% (+75% relativo).
   - Em código e em Python, ela **dobra** (26% → 52% e 21% → 45%).
   - Até em prosa inglesa, onde no early exit ela não ajudava, sobe +5 a +8 pontos.
   - A diferença em relação aos experimentos anteriores é o professor em todo token, porque o verificador vê tudo; antes, a memória só aprendia quando o córtex acordava.
2. **Mais memória ajuda.** O kNN 1024 supera o 256 em quase todo texto: com professor em todo token, 256 consultas cobrem só os últimos 256 tokens. O custo continua desprezível (~7 MB, e a busca custa <1% da cabeça).
3. **O MYELIN ganha com folga do cache neural.**
   - Na camada 8: +16 pontos contra +7.
   - Na camada 17: +10 contra +2.
   - Combinar os dois não acrescenta nada além do kNN.
   - Corrigir o estado oculto, em vez da distribuição de tokens, é o que faz a diferença.
4. **O ganho é maior onde o rascunho é pior:** camadas rasas e textos estruturados. Na camada 17, o estático já acerta bem (46%), e a memória soma +10 pontos.
5. **O speedup absoluto continua modesto no 0,5B.**
   - Na média, 1,22× no melhor caso; 1,37× em código.
   - A memória acrescenta ~+6 a +14% de speedup sobre o rascunho estático.
   - O limite é o custo do rascunho, porque ele paga a cabeça inteira em todo token.
   - Com a cabeça proporcional à de um 7B, chega a **1,62× em código**, só com uma memória de 7 MB, sem treinar nada.

## Ressalvas

- **Teacher forcing:** a aceitação foi medida sobre texto real, e não sobre o texto gerado pelo próprio modelo. Na geração, o modelo costuma ser mais previsível (aceitação maior), mas também pode se repetir, o que favoreceria o kNN. Medir em texto gerado é o próximo teste de fidelidade.
- **Speedup estimado, não cronometrado:**
  - Com γ = 1, E = 1 + a é exato, sem supor independência entre as aceitações.
  - O modelo de custo supõe que verificar γ+1 tokens custa um passo das camadas k+1..L (regime limitado por memória).
  - Falta uma implementação real com KV cache, medida em tempo de relógio.
- **Atraso do professor:** na simulação, a memória aprende com o token t antes de rascunhar o t+1. Com γ = 1, que foi sempre o ótimo, é exatamente o que acontece numa implementação real. Com γ > 1, o professor chegaria com até γ tokens de atraso.
- O speedup "com cabeça de 7B" é uma extrapolação: supõe que a aceitação na mesma profundidade relativa se mantenha, como a concordância no Experimento 5.
- Rascunhadores treinados (EAGLE, Medusa, LayerSkip, Kangaroo) têm aceitação bem maior, mas exigem treino. O MYELIN não precisa de treino e poderia ser **somado** a eles: uma memória rápida por cima de qualquer rascunhador.
- Uma rodada, 3 textos por grupo, β e min_sim do kNN sem ajuste.

## O que isso significa

Este é o resultado mais forte do projeto:
- **uma memória de tamanho fixo, sem treinar nada, ensinada pelo verificador que a decodificação especulativa já roda de graça**, aumenta a aceitação do rascunho em 15 de 15 textos, até o dobro em código;
- a qualidade da saída continua idêntica à do modelo completo.

A promessa original do MYELIN, acelerar sem perder qualidade, volta por este caminho. No 0,5B o ganho de velocidade é modesto, mas o mecanismo funciona.

## Próximos passos

1. **Revisão externa** do desenho e dos resultados.
2. **Fidelidade:** medir no texto gerado pelo próprio modelo (com amostragem, para evitar laços).
3. **Implementação real** da auto-especulação com KV cache, cronometrada na GPU.
4. **Modelo maior** (1.5B com tradutor amplo), onde a cabeça pesa menos.
5. **MYELIN por cima de um rascunhador treinado** (ex.: uma cabeça tipo EAGLE ou as saídas do LayerSkip).
