"""
Baixa e organiza o dataset ICL Ground Robot (Saeedi et al., ICRA 2019)
em ./dados/icl_ground_robot/.

Uso (a partir da raiz do projeto edge-depth-robotics-cv-study):

    uv run python codigo-treinamento/baixar_dados.py
    # ou, para customizar:
    uv run python codigo-treinamento/baixar_dados.py --data_root ./dados/icl_ground_robot

Fonte: https://peringlab.org/lmdata/
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# URLs oficiais (versão limpa, sem _noise). 1600 frames RGB-D em PNG por cena.
SCENES = {
    "deer":    "https://www.doc.ic.ac.uk/~wl208/lmdata/deer_robot.zip",
    "diamond": "https://www.doc.ic.ac.uk/~wl208/lmdata/diamond_ground_robot.zip",
}

# Trajetórias ground-truth (formato ViSim, ~kB).
GT_POSES = {
    "deer":    "https://www.doc.ic.ac.uk/~wl208/lmdata/deer_ground_robot.gt",
    "diamond": "https://www.doc.ic.ac.uk/~wl208/lmdata/diamond_ground_robot.gt",
}

# Default aponta para ./dados/icl_ground_robot a partir da raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dados" / "icl_ground_robot"


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} TB"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] já existe: {dest.name} ({human_size(dest.stat().st_size)})")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"  [get ] {url}")

    with urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                pct = 100 * got / total
                print(f"\r        {human_size(got)} / {human_size(total)} ({pct:5.1f}%)",
                      end="", flush=True)
        print()
    tmp.rename(dest)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [unzip] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for i, m in enumerate(members, 1):
            zf.extract(m, out_dir)
            if i % 200 == 0 or i == len(members):
                print(f"\r         {i}/{len(members)} arquivos", end="", flush=True)
    print()


def inspect_layout(scene_dir: Path) -> None:
    pngs = sorted(scene_dir.rglob("*.png"))
    print(f"  [info ] {scene_dir.name}: {len(pngs)} PNGs no total")
    if pngs:
        print("         primeiros nomes:")
        for s in [p.relative_to(scene_dir).as_posix() for p in pngs[:6]]:
            print(f"           {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help=f"Default: {DEFAULT_DATA_ROOT}")
    ap.add_argument("--keep_zips", action="store_true",
                    help="Manter os .zip baixados depois de extrair (default: apaga).")
    ap.add_argument("--scenes", nargs="+", choices=list(SCENES), default=list(SCENES))
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve()
    raw_dir = root / "_raw_zips"
    root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ICL] data_root = {root}")

    for scene in args.scenes:
        print(f"\n=== Cena: {scene} ===")
        zip_path = raw_dir / f"{scene}.zip"
        download(SCENES[scene], zip_path)

        gt_path = root / scene / "trajectory.gt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            download(GT_POSES[scene], gt_path)
        except Exception as e:
            print(f"  [warn] não baixou trajectory.gt: {e}")

        frames_dir = root / scene / "frames"
        if frames_dir.exists() and any(frames_dir.iterdir()):
            print(f"  [skip] extração já existe em {frames_dir}")
        else:
            extract_zip(zip_path, frames_dir)

        inspect_layout(frames_dir)

        if not args.keep_zips:
            try:
                zip_path.unlink()
                print(f"  [clean] removido {zip_path.name}")
            except OSError:
                pass

    print(f"\n[OK] Pronto. Use ICLGroundRobotDataset(root='{root}', ...) do dataloader.py")


if __name__ == "__main__":
    main()