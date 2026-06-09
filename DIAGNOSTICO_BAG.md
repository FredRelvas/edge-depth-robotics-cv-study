# Diagnóstico do rosbag — `experimento_20260609_211834`

Análise do primeiro bag real recebido (`dados/experimento_20260609_211834_0.db3`, 5,8 GB,
164 s, ROS 2 Humble / SQLite3). Gerado por `validacao/inspecionar_bag.py` + amostragem de
frames ao longo do bag.

**Resumo:** o bag tem o baseline (depth) e o ground truth (LiDAR) válidos, mas **não fecha
métrica como está** por causa de 3 problemas — dois exigem **regravar** (lado do robô),
um é calibração. O mais grave: a **predição da IA não está funcionando** (saída constante).

---

## O que veio no bag

| Tópico | Tipo / formato | Taxa | Estado |
|--------|----------------|------|--------|
| `/camera/camera/color/image_raw` | Image rgb8 **640×480**, frame `camera_color_optical_frame` | 15 Hz | ✅ ok |
| `/camera/camera/depth/image_rect_raw` | Image 16UC1 (mm) **640×480**, frame `camera_depth_optical_frame` | 15 Hz | ✅ ok — **baseline** |
| `/robot4/ia/depth_map` | Image 32FC1 **640×192**, frame `camera_color_optical_frame` | 14 Hz | ❌ **morto** (constante) |
| `/robot4/ia/depth_map/colorized` | Image bgr8 640×192 | 14 Hz | (visualização) |
| `/robot4/scan` | LaserScan 1080 feixes, 360°, range 0,15–12 m, 32% inválido | 8 Hz | ✅ ok — **ground truth** |
| `/robot4/tf`, `/robot4/tf_static` | árvore de frames do TB4 | 30 Hz / latched | ⚠️ ver problema 3 |
| `/robot4/odom`, `/robot4/imu` | odometria / IMU | 20 / 33 Hz | (não usados na métrica) |

> Observação: a câmera é uma **Intel RealSense** (namespace `/camera/camera/...`), não a
> OAK-D que o plano original assumia. O baseline passa a ser a depth da RealSense — o que é
> até melhor. Mas isso muda os nomes de tópico e a calibração (problemas 2 e 3).

---

## Problema 1 — A predição da IA está morta (BLOQUEADOR)

`/robot4/ia/depth_map` sai praticamente **constante** em todo o bag. Amostrando 5 frames
espalhados (0, 583, 1166, 1749, 2331):

```
frame    0: min=0.500 max=0.510 mean=0.504 std=0.0012
frame  583: min=0.500 max=0.508 mean=0.504 std=0.0011
frame 1166: min=0.499 max=0.510 mean=0.504 std=0.0013
frame 1749: min=0.499 max=0.510 mean=0.504 std=0.0014
frame 2331: min=0.501 max=0.509 mean=0.505 std=0.0013
```

O desvio-padrão na imagem inteira é ~0,001 → a saída é um **campo uniforme ~0,5** (cinza
chapado), sem nenhuma estrutura de cena.

**Importante:** isto **não é** o bug de normalização min-max que prevíamos. Normalização
[0,1] deixaria o span ≈ 1,0 (min→0, max→1). Um span de **0,01** indica que o **próprio
modelo está emitindo saída constante** — não é pós-processamento.

Causas prováveis (verificar nessa ordem):
- pesos do modelo **não carregados** (caminho do `.pt`/engine errado, fallback silencioso);
- o frame de entrada **não está chegando** na rede (o nó da IA assina o tópico da OAK/antigo
  e não a color da RealSense `/camera/camera/color/image_raw`);
- inferência falhando e devolvendo um **placeholder** (tensor zerado → ativação ~0,5);
- normalização/`sigmoid` aplicada sobre uma saída já degenerada.

**O que fazer (lado do robô, Frente 1):**
1. Confirmar nos logs do nó da IA que o modelo carregou (sem warning de fallback).
2. Garantir que o nó assina **`/camera/camera/color/image_raw`** (a câmera mudou para RealSense).
3. Logar `min/max/std` do tensor de saída por frame — tem que **variar** com a cena.
4. Publicar **depth bruto** (metros, float32), sem normalização min-max por frame. Manter o
   `colorized` só para vídeo.

Enquanto a saída da IA não variar com a cena, **o bag não serve para métrica de IA** — é o
único item que torna o run inútil por completo.

---

## Problema 2 — Sem `camera_info` (BLOQUEADOR para projeção)

Nenhum tópico de calibração foi gravado. Sem as **intrínsecas (`K`)** não é possível projetar
o LiDAR sobre os pixels da imagem, que é o passo central do cálculo de métrica.

**O que fazer (lado do robô):** incluir no `record_bag.sh`:
- `/camera/camera/color/camera_info`
- `/camera/camera/depth/camera_info`

(A RealSense publica ambos automaticamente.) Alternativa de contorno: enviar o **modelo exato
da RealSense** (ex. D435/D435i) para usarmos as intrínsecas de fábrica — menos preciso, mas
desbloqueia o desenvolvimento.

---

## Problema 3 — Extrínseca LiDAR→câmera ausente na TF (CALIBRAÇÃO)

A árvore `tf_static` traz os frames do TB4 (`base_link`, `rplidar_link`) e os da **OAK-D**
(`oakd_*`), mas **não** os frames da **RealSense** (`camera_color_optical_frame`,
`camera_depth_optical_frame`). Ou seja: a RealSense foi montada fisicamente, mas o transform
dela para `rplidar_link`/`base_link` **não é publicado** → a extrínseca **LiDAR → RealSense é
desconhecida**, e não dá para derivá-la do TF.

**O que fazer:**
- Publicar um `static_transform_publisher` de `rplidar_link` (ou `base_link`) para
  `camera_link` da RealSense — mesmo que a partir de **medição manual** do ponto de montagem; ou
- Fazer uma **calibração extrínseca** LiDAR↔câmera (mais preciso, recomendado para métrica a
  nível de pixel); ou
- No mínimo, me passar as **medidas físicas** (posição e orientação da RealSense em relação ao
  LiDAR) para montarmos a matriz 4×4 e validar com sanity check de reprojeção.

---

## Pendências menores (não bloqueiam, mas confirmar)

- **Alinhamento depth↔color.** A depth está em `camera_depth_optical_frame` e a color/IA em
  `camera_color_optical_frame`. Confirmar que a RealSense foi lançada com **`align_depth:=true`**
  — caso contrário, depth e IA não casam pixel-a-pixel.
- **Resolução da IA.** A IA sai em **640×192** (resize do color 640×480 — formato típico do
  **Monodepth2**, provavelmente o modelo deste run). No pipeline offline o `K` será ajustado
  para o crop/resize ao amostrar; sem problema, só registrar qual modelo gerou cada run.

---

## Checklist para o próximo bag

- [ ] IA publicando depth **que varia com a cena** (validar `std` por frame nos logs).
- [ ] IA recebendo **`/camera/camera/color/image_raw`** como entrada.
- [ ] Depth da IA em **metros, float32, sem normalização** min-max.
- [ ] Gravar `/camera/camera/color/camera_info` **e** `/camera/camera/depth/camera_info`.
- [ ] RealSense com **`align_depth:=true`**.
- [ ] Publicar/medir a **extrínseca LiDAR → RealSense** (TF estática ou medição física).
- [ ] (mantém) `scan`, `tf`/`tf_static`, `odom` — já estão corretos.

Com esses itens resolvidos, o pipeline offline (`validacao/`) calcula as métricas direto —
basta rodar `inspecionar_bag.py` no bag novo para reconfirmar e depois `avaliar.py`.
