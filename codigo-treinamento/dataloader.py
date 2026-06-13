"""
Dataloader do ICL Ground Robot para o edge-depth-robotics-cv-study.

Implementa o split 70/10/20 sequencial por cena conforme Vizzotto et al.
(RITA 2025), com faixa válida de profundidade de 1mm a 10m.

Uso típico no loop de treino:

    from dataloader import build_icl_dataloaders

    loaders = build_icl_dataloaders(
        scenes=("deer", "diamond"),
        image_size=256,           # ZoeDepth DPT_SwinV2_T_256
        batch_size=8,
        num_workers=4,
    )
    for batch in loaders["train"]:
        rgb   = batch["rgb"].cuda()      # [B, 3, 256, 256]
        depth = batch["depth"].cuda()    # [B, 1, 256, 256], em metros
        valid = batch["valid"].cuda()    # [B, 1, 256, 256], máscara
        ...

Smoke test (rode da raiz do projeto):

    uv run python codigo-treinamento/dataloader.py --scene deer
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constantes do paper (Vizzotto et al., RITA 2025)
# ---------------------------------------------------------------------------

MIN_DEPTH_M = 0.001   # 1 mm
MAX_DEPTH_M = 10.0    # 10 m
FRAMES_PER_SCENE = 1600
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

# Depth PNG 16-bit -> metros. O ICL Ground Robot do Pering Lab usa
# escala 1000 (1 unidade = 1 mm), confirmado empiricamente: o primeiro frame
# tem max=6694 (uint16), o que corresponde a 6.7m — coerente com o robô vendo
# uma parede no fundo de um quarto residencial.
DEFAULT_DEPTH_SCALE = 1000.0

# Default aponta para ./dados/icl_ground_robot a partir da raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dados" / "icl_ground_robot"


# ---------------------------------------------------------------------------
# Pareamento RGB <-> depth (detecta layout automaticamente)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(\d+)")


def _frame_id(name: str) -> Optional[int]:
    """Extrai o último número do nome do arquivo (índice do frame)."""
    matches = _NUM_RE.findall(name)
    return int(matches[-1]) if matches else None


def _find_subdir(parent: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Acha a primeira subpasta de `parent` cujo nome casa com algum candidato (case-insensitive)."""
    if not parent.is_dir():
        return None
    by_name = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for c in candidates:
        if c.lower() in by_name:
            return by_name[c.lower()]
    return None


def _frames_subdir(d: Path) -> Path:
    """No layout EuRoC/ASL os PNGs ficam em <d>/data/. Se existir, desce."""
    data_dir = d / "data"
    return data_dir if data_dir.is_dir() else d


def _pair_frames(frames_dir: Path) -> List[Tuple[int, Path, Path]]:
    """
    Pareia frames RGB e depth do ICL Ground Robot.

    O ZIP do Pering Lab extrai com o layout EuRoC/ASL (típico de SLAM):

        frames_dir/
            <scene>_robot/             (ex.: 'deer_robot')
                cam0/
                    data/              ← 1600 PNGs RGB
                    data.csv           (timestamps)
                depth0/
                    data/              ← 1600 PNGs de profundidade
                    data.csv
                cameraInfo.txt
                poses.gt

    A função desce os níveis necessários pra encontrar as pastas data/.
    """
    # 1) Localiza a pasta-base que tem cam0/ e depth0/ (pode ser frames_dir
    #    diretamente ou frames_dir/<algo>_robot/).
    candidates = [frames_dir] + [p for p in frames_dir.iterdir() if p.is_dir()]
    base: Optional[Path] = None
    rgb_dir: Optional[Path] = None
    dep_dir: Optional[Path] = None
    for c in candidates:
        r = _find_subdir(c, ("cam0", "rgb", "rgb0", "image_0", "images"))
        d = _find_subdir(c, ("depth0", "depth", "depth_0", "depths"))
        if r is not None and d is not None:
            base = c
            # No layout EuRoC os PNGs ficam em cam0/data/ e depth0/data/.
            rgb_dir = _frames_subdir(r)
            dep_dir = _frames_subdir(d)
            break

    if rgb_dir is None or dep_dir is None:
        raise RuntimeError(
            f"Não encontrei subpastas cam0/ e depth0/ em {frames_dir}. "
            f"Conteúdo: {sorted(p.name for p in frames_dir.rglob('*') if p.is_dir())[:10]}"
        )

    # 2) Indexa pelos nomes. No EuRoC os arquivos costumam ser timestamps grandes
    #    (ex.: 1403715273262142976.png). Usamos o número extraído como id ordenável.
    rgbs: Dict[int, Path] = {}
    depths: Dict[int, Path] = {}
    for p in sorted(rgb_dir.glob("*.png")):
        fid = _frame_id(p.name)
        if fid is not None:
            rgbs[fid] = p
    for p in sorted(dep_dir.glob("*.png")):
        fid = _frame_id(p.name)
        if fid is not None:
            depths[fid] = p

    common = sorted(set(rgbs) & set(depths))
    if not common:
        # Fallback: se RGB e depth foram nomeados com timestamps diferentes,
        # pareia por posição ordenada (mesma contagem de frames).
        rgb_sorted = sorted(rgbs.items())
        dep_sorted = sorted(depths.items())
        if len(rgb_sorted) == len(dep_sorted) and rgb_sorted:
            return [(i, rgb_sorted[i][1], dep_sorted[i][1])
                    for i in range(len(rgb_sorted))]
        raise RuntimeError(
            f"Não consegui parear RGB e depth em {base}. "
            f"Achei {len(rgbs)} RGB em {rgb_dir} e {len(depths)} depth em {dep_dir}."
        )

    return [(fid, rgbs[fid], depths[fid]) for fid in common]


# ---------------------------------------------------------------------------
# Split sequencial 70/10/20 por cena
# ---------------------------------------------------------------------------

def _split_indices(n: int, split: str) -> Tuple[int, int]:
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val   = int(round(n * SPLIT_RATIOS["val"]))
    if split == "train":
        return 0, n_train
    if split == "val":
        return n_train, n_train + n_val
    if split == "test":
        return n_train + n_val, n
    raise ValueError(f"split inválido: {split!r}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ICLGroundRobotDataset(Dataset):
    """
    Args:
        root:        diretório raiz com as cenas (default: ./dados/icl_ground_robot).
        scene:       'deer' | 'diamond' | iterável das duas (concatena).
        split:       'train' | 'val' | 'test' (sequencial por cena, 70/10/20).
        image_size:  inteiro (lado quadrado) ou (H, W). Default 256.
        depth_scale: divisor pra converter int16 do PNG em metros.
        min_depth, max_depth: faixa válida em metros (paper: 0.001 .. 10).
        hflip_prob:  probabilidade de horizontal flip (paper: 0.5 no treino).
        augment:     True aplica flip; None = automático (True só em train).
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_ROOT,
        scene: str | Sequence[str] = ("deer", "diamond"),
        split: str = "train",
        image_size: int | Tuple[int, int] = 256,
        depth_scale: float = DEFAULT_DEPTH_SCALE,
        min_depth: float = MIN_DEPTH_M,
        max_depth: float = MAX_DEPTH_M,
        hflip_prob: float = 0.5,
        augment: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hflip_prob = hflip_prob
        self.augment = (split == "train") if augment is None else augment

        if isinstance(image_size, int):
            self.image_size: Tuple[int, int] = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)  # type: ignore[assignment]

        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        self.scenes = scenes

        self.samples: List[Dict] = []
        for sc in scenes:
            scene_dir = self.root / sc / "frames"
            if not scene_dir.exists():
                raise FileNotFoundError(
                    f"Não encontrei {scene_dir}. Rode primeiro: "
                    f"uv run python codigo-treinamento/baixar_dados.py"
                )
            pairs = _pair_frames(scene_dir)
            n = len(pairs)
            if n != FRAMES_PER_SCENE:
                print(f"[ICL][warn] cena {sc!r}: esperava {FRAMES_PER_SCENE} frames, "
                      f"encontrei {n}. Split aplicado proporcionalmente.")
            start, end = _split_indices(n, split)
            for fid, rgb_path, depth_path in pairs[start:end]:
                self.samples.append({
                    "scene": sc,
                    "frame_index": fid,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                })

        if not self.samples:
            raise RuntimeError(f"Split {split!r} ficou vazio para cenas {scenes}.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize(
            (self.image_size[1], self.image_size[0]),  # PIL: (W, H)
            resample=Image.BILINEAR,
        )
        return np.array(img, dtype=np.float32) / 255.0

    def _load_depth(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode not in ("I", "I;16", "I;16B", "F"):
            img = img.convert("I")
        depth = np.array(img, dtype=np.float32) / self.depth_scale  # -> metros
        if depth.shape != self.image_size:
            d_img = Image.fromarray(depth, mode="F").resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.NEAREST,  # nearest pra não interpolar profundidade
            )
            depth = np.array(d_img, dtype=np.float32)
        return depth

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rgb = self._load_rgb(s["rgb_path"])
        depth = self._load_depth(s["depth_path"])

        if self.augment and self.hflip_prob > 0 and np.random.rand() < self.hflip_prob:
            rgb = rgb[:, ::-1, :].copy()
            depth = depth[:, ::-1].copy()

        valid = (depth >= self.min_depth) & (depth <= self.max_depth)

        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
            "depth": torch.from_numpy(depth).unsqueeze(0).contiguous(),
            "valid": torch.from_numpy(valid).unsqueeze(0).contiguous(),
            "scene": s["scene"],
            "frame_index": s["frame_index"],
            "rgb_path": s["rgb_path"],
            "depth_path": s["depth_path"],
        }


# ---------------------------------------------------------------------------
# Helper: 3 DataLoaders prontos para treino
# ---------------------------------------------------------------------------

def build_icl_dataloaders(
    root: str | Path = DEFAULT_DATA_ROOT,
    scenes: Sequence[str] = ("deer", "diamond"),
    image_size: int | Tuple[int, int] = 256,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> Dict[str, DataLoader]:
    """
    Cria 'train', 'val' e 'test' com as convenções do paper.

    Args:
        batch_size:      batch do TREINO.
        eval_batch_size: batch do val/test. Se None, usa o mesmo do treino.
                         Para ZoeDepth com fine-tune completo do BEiT em GPU
                         24GB, recomenda-se eval_batch_size=1 — o BEiT-Large
                         ainda consome muita memória nos act_postprocess (o
                         torch.cat dobra os feature maps temporariamente),
                         então batch pequeno no eval é a forma mais robusta
                         de evitar OOM logo após o treino.
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size

    common = dict(root=root, scene=scenes, image_size=image_size, depth_scale=depth_scale)
    train_ds = ICLGroundRobotDataset(**common, split="train", augment=True)
    val_ds   = ICLGroundRobotDataset(**common, split="val",   augment=False)
    test_ds  = ICLGroundRobotDataset(**common, split="test",  augment=False)

    print(f"[ICL] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"(cenas={list(scenes)})  "
          f"batch_train={batch_size}  batch_eval={eval_batch_size}")

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  drop_last=True,  **common_kwargs),
        "val":   DataLoader(val_ds,   batch_size=eval_batch_size,
                            shuffle=False, drop_last=False, **common_kwargs),
        "test":  DataLoader(test_ds,  batch_size=eval_batch_size,
                            shuffle=False, drop_last=False, **common_kwargs),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--scene", type=str, default="deer", choices=["deer", "diamond"])
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        ds = ICLGroundRobotDataset(
            root=args.root, scene=args.scene, split=split,
            image_size=args.image_size, augment=False,
        )
        sample = ds[0]
        d = sample["depth"]
        d_valid = d[sample["valid"]]
        print(
            f"[{args.scene:7s}/{split:5s}]  N={len(ds):4d}  "
            f"rgb={tuple(sample['rgb'].shape)}  depth={tuple(d.shape)}  "
            f"depth_min={d_valid.min().item():.3f}m  "
            f"depth_max={d_valid.max().item():.3f}m  "
            f"valid_frac={sample['valid'].float().mean().item():.3f}"
        )