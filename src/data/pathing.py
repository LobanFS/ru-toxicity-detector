from pathlib import Path

def project_root():
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists():
            return parent
    return here.parents[-1]

BASE = project_root()
DATA = BASE / "data"