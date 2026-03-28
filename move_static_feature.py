from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

old_path = ROOT / "results" / "static_features_12d.csv"
new_dir = ROOT / "data" / "processed"
new_path = new_dir / "static_features_12d.csv"

new_dir.mkdir(parents=True, exist_ok=True)

if old_path.exists():
    if not new_path.exists():
        shutil.move(str(old_path), str(new_path))
        print(f"MOVED: {old_path} → {new_path}")
    else:
        print(f"TARGET EXISTS: {new_path}")
else:
    print(f"NOT FOUND: {old_path}")
    