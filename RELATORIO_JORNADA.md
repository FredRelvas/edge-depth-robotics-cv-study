# Relatório de jornada — por que o artigo mudou tanto (e o que falta)

> **Para o time.** Este documento é o "diário de bordo" das descobertas que fizemos
> revisando os dados e o artigo. Ele explica **por que a versão do artigo mudou** de
> uma hora para outra e **o que ainda precisamos** para fechar. Ele **não** é o
> artigo: o artigo (`artigo/artigo.tex`) é escrito como um paper limpo e atemporal,
> sem este "changelog". Esta narrativa fica só aqui.

## 1. De onde partimos

A primeira versão do artigo afirmava, como resultado principal, que os modelos de IA
**superavam o sensor nativo (baseline) em 60–70%** no erro relativo. Era a leitura
direta dos números do relatório de implantação.

## 2. O primeiro rosbag estava quebrado

Ao inspecionar o primeiro bag recebido (`dados/experimento_20260609_211834`, de
9/jun), encontramos três problemas — documentados em detalhe no `DIAGNOSTICO_BAG.md`:

1. **A IA estava "morta":** `/robot4/ia/depth_map` saía praticamente **constante**
   (~0,5, desvio ~0,001 na imagem toda) em todos os frames. Não era normalização —
   o modelo não estava inferindo. (Corrigido depois: ver §4.4 do relatório, sanity
   de `std` no boot.)
2. **Sem `camera_info`:** sem as intrínsecas, não há projeção do LiDAR.
3. **A câmera tinha mudado:** de OAK-D para **Intel RealSense D415** (namespace
   `/camera/...`), com o baseline passando a ser a depth da RealSense.

Esse bag de 9/jun é o **antigo/quebrado**. Os resultados do relatório vieram dos bags
**bons de 10/jun** (já com a IA inferindo e `camera_info` gravado).

## 3. A descoberta principal — a comparação estava **assimétrica**

Mesmo sem os bags bons, conseguimos investigar o baseline no bag antigo (porque o
baseline×LiDAR **não depende da IA**). Criamos `validacao/diag_baseline.py`, que
compara o baseline contra o LiDAR **cru** vs **com o mesmo alinhamento por mediana
que a IA recebe**. Resultado:

| Baseline | Abs Rel | RMSE | δ<1.25 |
|---|---|---|---|
| **Cru** (como estava reportado) | 4,305 | 2,123 m | 0,140 |
| **Alinhado** (mesmo tratamento da IA) | **0,236** | 1,029 m | **0,722** |

Ou seja: **o alinhamento sozinho explica quase todo o "buraco" do baseline** (Abs Rel
melhora ~18×). O relatório comparava **IA-alinhada vs baseline-NÃO-alinhado** — uma
luta desigual. Quando os dois passam pelo mesmo alinhamento, o baseline fica
**competitivo** (e, neste bag, até melhor).

Também confirmamos que o **`align_depth` estava desligado** (a depth está em
`camera_depth_optical_frame`, sem o tópico `aligned_depth_to_color`), o que adiciona
uma penalidade extra **só ao baseline**.

> Observação importante: os "80% de zeros" **não** eram a causa do baseline ruim —
> nosso pipeline já descarta `oak==0` antes da métrica. A causa real é a assimetria
> de alinhamento + `align_depth` off + extrínseca placeholder.

## 4. Decisão metodológica e o que mudou no artigo

**Decisão:** o **alinhamento de escala passa a ser o padrão, aplicado igualmente à IA
e ao baseline**. O artigo não discute "com vs sem alinhamento" — parte de tudo
alinhado. Com isso, a comparação é justa por construção.

Principais mudanças no `artigo.tex`:
- **Resultado central reenquadrado** para *"precisão **competitiva** com o sensor
  nativo"* (em vez de "superamos em 60–70%").
- **§4.3** agora explica direito o que são as matrizes **intrínseca** ($K$) e
  **extrínseca** ($T$).
- **§4.4** explicita que o alinhamento é aplicado **aos dois** (padrão), explica por
  que alinhar e por que dois métodos (mediana p/ Monodepth2, afim p/ DAv2).
- **Quadro de métricas** com fórmula + interpretação de cada uma.
- Limitações honestas (`align_depth` off, extrínseca placeholder, ambiente único) e
  justificativa técnica para o **ZoeDepth** não ter sido avaliado (hardware +
  quantização).
- Links oficiais (repos, HuggingFace, Kaggle) e placeholders das figuras.

## 5. PENDÊNCIAS — o que precisamos do time

Bloqueiam o fechamento do artigo. Em ordem:

- [ ] **Bags bons das 3 runs de 10/jun** (com o João) — bloqueiam tudo abaixo.
- [ ] **Recalcular as métricas do BASELINE COM alinhamento**, nas 3 runs, para
      preencher os `[?]` da Tabela 4 (head-to-head). Ferramenta pronta:
      `validacao/diag_baseline.py` (já calcula o baseline alinhado).
- [ ] **Confirmar as 7 métricas do baseline** (hoje só temos 3: Abs Rel, RMSE, δ₁).
- [ ] **(Idealmente) regravar com `align_depth:=true`** + extrínseca medida (fita +
      nível, ou hand-eye). Se não der tempo, fica declarado como limitação.
- [ ] **Figuras 1, 3 e 4** (LiDAR projetado no RGB; painel RGB|IA|baseline;
      predito×GT) — geradas a partir dos bags bons.
- [ ] **Autores e afiliação** reais no `artigo.tex` (hoje há placeholders).
- [ ] **Escrever as análises/conclusões finais** derivadas dos números (marcadas como
      "[A PREENCHER amanhã]" no `.tex`).

## 6. Estado atual dos arquivos

- `artigo/artigo.tex` — atualizado, com placeholders `[?]` nas células do baseline e
  "[A PREENCHER amanhã]" nas análises numéricas. Compila no Overleaf.
- `artigo/artigo_overleaf.zip` — pronto para subir.
- `DIAGNOSTICO_BAG.md` — diagnóstico do bag quebrado (referência).
- `validacao/diag_baseline.py` — script de verificação do baseline (rodar nos bags
  bons).
