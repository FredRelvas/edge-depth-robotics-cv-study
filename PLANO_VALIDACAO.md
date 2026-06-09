# Plano de Validação no TurtleBot4

Validação prática dos modelos de estimativa de profundidade replicados neste
repositório, rodando num **TurtleBot4 real** e comparando suas predições com a
profundidade do **estéreo nativo da OAK-D** (baseline) e com a distância medida
pelo **LiDAR** (ground truth).

> **Pergunta científica:** para cada modelo, o depth estimado por deep learning
> supera a precisão do hardware estéreo nativo do robô?

---

## Visão geral

- **Modelos validados:** ZoeDepth (supervisionado, métrico), Monodepth2
  (autossupervisionado, escala ambígua) e Depth-Anything ViT-S (relativo) — este
  já vem integrado ao repo do robô como referência extra.
- **Foco:** cada modelo **vs. baseline (OAK-D) vs. LiDAR** — *não* modelo contra
  modelo.
- **Coleta:** **3 runs independentes**, um modelo por vez rodando ao vivo no robô
  (nunca simultâneos). Cada run grava seu próprio rosbag com a predição da IA, a
  profundidade da OAK-D e o scan do LiDAR.
- **Cálculo:** offline, projetando o LiDAR sobre os pixels da imagem e montando a
  tabela `y_lidar / y_oak / y_ia` para calcular as métricas.

O setup embarcado vive em repositório separado (**JoaoGChv/TB4**): ROS 2 Humble,
namespace `/robot4`, OAK-D Pro, Jetson Orin Nano, pacote `tb4_depth_estimator`.

---

## Arquitetura: coleta em tempo real, métricas offline

Para não sobrecarregar a Jetson durante a inferência, a validação é dividida em
duas fases que separam a carga de inferência da carga de cálculo.

**Fase 1 — Coleta (tempo real, no robô).** O modelo roda embarcado na Jetson
(TensorRT) recebendo os frames da OAK-D e gerando o depth ao vivo no maior FPS
possível. Em paralelo, um nó gravador passivo (baixa CPU) salva um rosbag com os
dados sincronizados do instante: RGB, predição da IA, profundidade da OAK-D, scan
do LiDAR e as transformações (TF). O FPS/latência de cada modelo sai do log do nó.

**Fase 2 — Cálculo (offline, no PC/`uv`).** Um script lê o rosbag, sincroniza os
frames por timestamp, projeta os feixes do LiDAR sobre os pixels da imagem via
matrizes de transformação, monta a tabela de validação e calcula as métricas.

> **Por que separar:** se a projeção do LiDAR e o cálculo de erro rodassem no
> mesmo nó da IA ao vivo, a CPU da Jetson gargalaria, derrubando o FPS e quebrando
> a sincronia das mensagens do ROS 2.

---

## Frentes de trabalho

### Frente 1 — Preparar o robô (repo TB4)

| # | Tarefa |
|---|--------|
| 1.1 | Habilitar o **estéreo da OAK-D**: publicar `/robot4/stereo/depth` + `camera_info`, alinhado ao frame RGB (hoje a config é RGB-only, `i_pipeline_type: RGB`). |
| 1.2 | Corrigir o `depth_node.py` para publicar depth **RAW** (float32), **sem a normalização min-max por frame** — caso contrário o `y_ia` gravado perde a escala métrica. Manter a versão colorizada apenas para visualização/vídeo. |
| 1.3 | (opcional) Aumentar a resolução do RGB preview — 250×250 é baixo para projeção precisa do LiDAR. |
| 1.4 | Ajustar o `record_bag.sh` para gravar, por run: RGB, `ia/depth_map` (raw), `stereo/depth`, ambos `camera_info`, `scan`, `tf`/`tf_static`, `odom`. |
| 1.5 | Exportar ZoeDepth e Monodepth2 para ONNX → TensorRT e adicionar como backends do `depth_node` (Depth-Anything já existe). |

### Frente 2 — Coleta (3 runs)

| # | Tarefa |
|---|--------|
| 2.1 | Para cada modelo: subir o backend na Jetson → `check_system.sh` → `undock.sh` → gravar 2–3 min de trajetória **lenta / com paradas** (mitiga o skew temporal do LiDAR 2D). |
| 2.2 | (opcional) Renderizar o `depth_map/colorized` de cada run em vídeo para comparação visual posterior. |

### Frente 3 — Calibração

| # | Tarefa |
|---|--------|
| 3.1 | Extrair as intrínsecas da câmera a partir do `camera_info` gravado. |
| 3.2 | Obter a extrínseca **LiDAR → câmera**: partir da TF do URDF, validar com sanity check de reprojeção; calibrar de fato se o erro for grande. |

### Frente 4 — Pipeline offline (neste repo, ambiente `uv`)

| # | Tarefa | Detalhe |
|---|--------|---------|
| 4.1 | Loader de rosbag via lib `rosbags` (lê `.mcap`/`.db3` sem ROS instalado), com sincronização aproximada por timestamp. | — |
| 4.2 | `projetar_lidar.py`: projeta o scan 2D sobre os pixels → ground truth esparso (faixa horizontal única). | — |
| 4.3 | Montar a tabela por run: `y_lidar / y_oak / y_ia` nos mesmos pixels; **unificar unidades para metros** (OAK-D normalmente em mm / uint16). | — |
| 4.4 | Alinhar a escala conforme o modelo do run. | Zoe: direto · Mono: mediana · Depth-Anything: afim (scale+shift) · OAK-D: direto (métrico) |
| 4.5 | Limpar pontos inválidos (LiDAR infinito, OAK-D zero, fora do FOV) e calcular RMSE / Abs Rel / δ reusando `metricas/metricas.py`. | — |

---

## Dependências

```
F1 ──> F2 ──┬──> F3
            └──> F4
```

Caminho crítico para o resultado científico: **F1 → F2 → F4**. A Frente 3 trava a
precisão da projeção. A Frente 4 roda na máquina de desenvolvimento e não depende
do robô (bom ponto de partida em paralelo à preparação do hardware).

---

## Riscos conhecidos

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| **LiDAR do TB4 é 2D** (RPLIDAR) | Ground truth esparso, restrito a uma faixa horizontal de pixels | Documentar na metodologia; agregar muitos frames para significância estatística (abordagem aceita, ex. KITTI) |
| **Calibração extrínseca** LiDAR↔OAK-D | TF nominal do URDF pode ser imprecisa para projeção a nível de pixel | Sanity check de reprojeção; calibrar se necessário |
| **Sincronização temporal** (robô em movimento) | LiDAR ~10 Hz vs câmera ~30 Hz nunca coincidem no mesmo ms | `ApproximateTime`; coletar com robô lento / parando |
| **Escala do Monodepth2 / Depth-Anything** | Profundidade sem escala métrica | Alinhamento por mediana (Mono) / afim (Depth-Anything) |
| **Domain gap** | Modelos treinados no ICL Ground Robot rodando em ambiente novo → métricas piores | Esperado — é justamente o que o experimento mede, não é bug |
| **Unidades** | OAK-D em mm, IA/LiDAR em metros | Unificar tudo para metros antes de montar a tabela |
| **Viés do LiDAR mascarado pelo alinhamento** | Usar o LiDAR como régua *e* gabarito faz um erro sistemático dele ser absorvido pelo alinhamento dos modelos relativos (some no Monodepth2/Depth-Anything, mas o ZoeDepth o expõe) → comparação assimétrica | Validar o LiDAR contra distância conhecida; reportar também a escala ancorada pela **altura da câmera** (independente do LiDAR) — ver abaixo |

## Duas fontes de escala (avaliação honesta)

Modelos relativos precisam de uma referência para virar metros. O pipeline da Frente 4
suporta duas, e vale reportar ambas:

- **Via LiDAR** (`--fonte-escala lidar`): ajusta a escala/forma contra o próprio LiDAR.
  Mede bem a **estrutura**, mas dá "ajuda" aos modelos relativos e **esconde viés
  sistemático** do LiDAR.
- **Via altura da câmera** (`--fonte-escala altura --altura-camera 0.395`): recupera a
  escala pelo **plano do chão** (RANSAC) + a altura medida da câmera ao chão (**39,5 cm**
  no nosso TB4), sem usar o LiDAR. O LiDAR vira **só gabarito** → erros dele aparecem
  honestamente. Aplica-se a ZoeDepth e Monodepth2; **não** ao Depth-Anything (2 graus de
  liberdade), que permanece no LiDAR ou exigiria a variante *Depth-Anything Metric*.

Demonstração (sintética, LiDAR com viés α=1,08): escala via LiDAR mascara o erro no
Monodepth2 (Abs Rel ≈ 0,001); escala via altura o revela (≈ 0,075), igual ao ZoeDepth.
