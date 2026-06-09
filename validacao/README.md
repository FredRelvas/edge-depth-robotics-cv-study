# validacao/ — Pipeline de validação offline (Frente 4)

Lê um rosbag gravado num run do TurtleBot4, projeta o LiDAR 2D sobre os pixels da
imagem e calcula as métricas de profundidade comparando **cada modelo** com o
**baseline (OAK-D)** e o **ground truth (LiDAR)**. Ver `PLANO_VALIDACAO.md` na raiz.

Como a predição da IA (`y_ia`) é gravada ao vivo no bag, este pipeline **não carrega
modelos nem faz inferência** — só lê arrays, projeta geometria e reusa
`metricas/metricas.py`.

## Módulos

| Arquivo | Papel |
|---------|-------|
| `leitor_rosbag.py` | Lê `.mcap`/`.db3` via lib `rosbags`; sincroniza scan e OAK ao frame da IA por timestamp |
| `geometria.py` | LaserScan→pontos, transform LiDAR→câmera, projeção pinhole (GT = Z), amostragem |
| `alinhamento.py` | Escala da predição: `direto` (ZoeDepth/OAK), `mediana` (Monodepth2), `afim_disparidade` (Depth-Anything) |
| `calibracao.py` | Carrega K + extrínseca 4×4 do YAML (fonte única de verdade) |
| `avaliacao.py` | Monta a tabela esparsa `y_lidar/y_oak/y_ia`, limpa inválidos, chama as métricas |
| `avaliar.py` | CLI |
| `gerar_bag_sintetico.py` | Gera um `.mcap` com geometria conhecida para testar sem o robô |
| `config/calibracao_exemplo.yaml` | Intrínseca + extrínseca de exemplo |

## Ao receber o bag real: inspecionar primeiro

Antes de avaliar, rode o inspetor — ele faz o raio-x do bag e sinaliza problemas
(ia normalizado, stereo em frame diferente do RGB, tópicos faltando, nomes diferentes):

```bash
uv run python validacao/inspecionar_bag.py <caminho_do_bag>
```

A partir do relatório decidimos as adaptações necessárias (override de nomes, projeção
no frame estéreo, extrínseca via TF) com base em evidência, não em suposição.

## Uso

```bash
# Avaliar um bag real de um run (um modelo por run):
uv run python validacao/avaliar.py \
    --bag caminho/do/run_zoedepth --modelo zoedepth \
    --calibracao validacao/config/calibracao_exemplo.yaml \
    --saida resultados/run_zoedepth.json
```

`--modelo` ∈ `{zoedepth, monodepth2, depthanything}` — seleciona o modo de alinhamento.

### Fonte de escala: LiDAR ou altura da câmera

Modelos relativos (Monodepth2, Depth-Anything) precisam de uma referência para virar
metros. Há duas formas, via `--fonte-escala`:

```bash
# (padrão) escala ajustada contra o próprio LiDAR
uv run python validacao/avaliar.py --bag <run> --modelo monodepth2 --fonte-escala lidar

# escala pela altura conhecida da câmera ao chão (independente do LiDAR)
uv run python validacao/avaliar.py --bag <run> --modelo monodepth2 \
    --fonte-escala altura --altura-camera 0.395
```

- **`lidar`** — usa o LiDAR como régua *e* como gabarito. Mede bem a estrutura, mas um
  erro sistemático do LiDAR é **absorvido** pelo alinhamento (fica escondido), e os
  modelos relativos ganham "ajuda" do gabarito.
- **`altura`** — recupera a escala pelo plano do chão (RANSAC) + altura conhecida, sem
  tocar no LiDAR. Assim o LiDAR vira **só gabarito** e qualquer erro dele aparece
  honestamente. Aplica-se a ZoeDepth (fica ≈ métrico) e Monodepth2 (1 grau de liberdade).
  **Não** fecha para o Depth-Anything (2 graus de liberdade: escala + shift) → cai
  automaticamente no LiDAR (o relatório mostra `escala via: lidar`).

Demonstração do viés (cena `parede_chao`, LiDAR com α=1.08): com escala via LiDAR o
Monodepth2 mascara o erro (Abs Rel ≈ 0,001), mas via altura o mesmo erro aparece
(≈ 0,075) — igual ao ZoeDepth, que o expõe nos dois casos.

## Teste sem o robô (dados sintéticos)

```bash
# Gera um bag sintético e valida o pipeline de ponta a ponta:
uv run python validacao/gerar_bag_sintetico.py --modelo zoedepth --out /tmp/sim_zoe
uv run python validacao/avaliar.py --bag /tmp/sim_zoe --modelo zoedepth
```

Com a cena sintética (ruído pequeno), o esperado é `Abs Rel ≈ 0`, `RMSE ≈ 0 m`,
`δ<1.25 ≈ 1.0` — tanto para a IA quanto para o OAK-D — confirmando que projeção,
amostragem, alinhamento e métricas estão corretos. Trocar `--modelo` por
`monodepth2`/`depthanything` exercita os outros alinhamentos.

Cena `parede_chao` (parede + chão a 0,395 m) para testar a ancoragem por altura — o
LiDAR horizontal só atinge a parede, e o chão (visível só para a câmera) fixa a escala:

```bash
uv run python validacao/gerar_bag_sintetico.py --modelo monodepth2 --cena parede_chao --out /tmp/pc
uv run python validacao/avaliar.py --bag /tmp/pc --modelo monodepth2 --fonte-escala altura --altura-camera 0.395
```

Sanity por módulo (sem bag):

```bash
uv run python validacao/geometria.py
uv run python validacao/alinhamento.py
```

## Contrato de tópicos (espelha o `record_bag.sh` do repo TB4)

| Tópico | Tipo | Uso |
|--------|------|-----|
| `/robot4/ia/depth_map` | Image 32FC1 | `y_ia` (bruto) |
| `/robot4/stereo/depth` | Image 16UC1 (mm) | `y_oak` baseline |
| `/robot4/scan` | LaserScan | LiDAR (GT esparso) |
| `/robot4/oakd/rgb/preview/camera_info` | CameraInfo | intrínseca K |

## Notas / limitações conhecidas

- O LiDAR do TB4 é **2D**: o GT é esparso, numa faixa horizontal de pixels. Agregar
  muitos frames para significância estatística.
- A **extrínseca** vem do arquivo de calibração (Frente 3). O bag sintético usa uma
  extrínseca conhecida; em bags reais, refine a calibração antes de confiar nas
  métricas a nível de pixel.
- A leitura de `tf_static` do bag para derivar a extrínseca automaticamente é uma
  evolução futura — hoje a extrínseca é fornecida pelo YAML.
