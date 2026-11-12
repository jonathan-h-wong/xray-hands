"""Download the mesh-only subset of ARCTIC (no images).

Fetches raw_seqs (MANO + object params), meta (object templates), splits_json,
and the MANO v1.2 model. Prompts for credentials so they never touch disk/history.

Run from the project root in an interactive terminal:
    .venv\\Scripts\\python scripts\\download_arctic.py
"""
import getpass
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCTIC_REPO = ROOT / "tools" / "arctic"
OUT = ROOT / "data" / "arctic"
DL = OUT / "downloads"

sys.path.insert(0, str(ARCTIC_REPO / "scripts_data"))
from download_data import download_data  # noqa: E402

MISC = [u for u in (ARCTIC_REPO / "bash/assets/urls/misc.txt").read_text().split()
        if "backgrounds.zip" not in u]


def ask(flag, label):
    if not os.environ.get(f"{flag}_USERNAME"):
        os.environ[f"{flag}_USERNAME"] = input(f"{label} username (email): ").strip()
    if not os.environ.get(f"{flag}_PASSWORD"):
        os.environ[f"{flag}_PASSWORD"] = getpass.getpass(f"{label} password: ")


def check_zip(p: Path):
    if not zipfile.is_zipfile(p):
        head = p.read_bytes()[:200]
        sys.exit(f"ERROR: {p.name} is not a zip (bad credentials or license not accepted?). "
                 f"First bytes: {head!r}")


def main():
    ask("ARCTIC", "ARCTIC")
    ask("MANO", "MANO")
    (DL / "data").mkdir(parents=True, exist_ok=True)

    misc_list = DL / "misc_meshes_only.txt"
    misc_list.write_text("\n".join(MISC))
    download_data(str(misc_list), str(DL / "data"), dry_run=False)
    download_data(str(ARCTIC_REPO / "bash/assets/urls/mano.txt"), str(DL), dry_run=False)

    zips = [DL / "data" / n for n in ("raw_seqs.zip", "meta.zip", "splits_json.zip")]
    zips.append(DL / "mano_v1_2.zip")
    for z in zips:
        check_zip(z)

    data_dir = OUT / "unpack" / "arctic_data" / "data"
    for z in zips[:3]:
        print(f"Unzipping {z.name} -> {data_dir}")
        zipfile.ZipFile(z).extractall(data_dir)
    mano_dir = OUT / "unpack" / "body_models"
    print(f"Unzipping mano_v1_2.zip -> {mano_dir}")
    zipfile.ZipFile(zips[3]).extractall(mano_dir)
    print("Done. ARCTIC mesh subset in", OUT / "unpack")


if __name__ == "__main__":
    main()
