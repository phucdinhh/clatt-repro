"""Config freeze before the test split is read (stage 7.0), without a git commit.

python -m src.freeze write   records the sha256 of every config file in configs/FROZEN.sha256
python -m src.freeze check   lists files that changed, were added or were removed since then
`src.run --final` refuses to start unless `check` is clean.
"""
import hashlib
import sys

from src.data import ROOT

CONFIGS = ROOT / "configs"
MANIFEST = CONFIGS / "FROZEN.sha256"


def config_files():
    return sorted(p for p in CONFIGS.rglob("*.yaml") if "experiments" not in p.parts)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write():
    MANIFEST.write_text("".join(f"{digest(p)}  {p.relative_to(ROOT)}\n" for p in config_files()))


def check():
    """Problems as strings; empty means the configs are exactly as frozen."""
    if not MANIFEST.exists():
        return [f"{MANIFEST.relative_to(ROOT)} missing: run python -m src.freeze write before --final"]
    frozen = dict(line.split("  ", 1)[::-1] for line in MANIFEST.read_text().splitlines() if line)
    now = {str(p.relative_to(ROOT)): digest(p) for p in config_files()}
    return ([f"changed: {f}" for f in frozen if f in now and now[f] != frozen[f]]
            + [f"removed: {f}" for f in frozen if f not in now]
            + [f"added: {f}" for f in now if f not in frozen])


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        write()
    problems = check()
    print("\n".join(problems) or "configs match the freeze")
    sys.exit(1 if problems else 0)
