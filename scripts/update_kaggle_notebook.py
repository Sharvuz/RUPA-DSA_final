"""Make the checked-in Kaggle notebook clone the RUPA-DSA v0 repo directly."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "RUPA_DSA_Kaggle.ipynb"


def lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

config = "".join(notebook["cells"][1]["source"])
config = config.replace(
    'REPO_URL = "https://github.com/lessiYin/DSANet.git"',
    'REPO_URL = "https://github.com/Sharvuz/RUPA-DSA-v0.git"',
)
notebook["cells"][1]["source"] = config.splitlines(keepends=True)

notebook["cells"][2]["source"] = lines(
    """
    ## 1. Clone RUPA-DSA v0 and prepare the GPU environment

    Kaggle clones the source directly from `Sharvuz/RUPA-DSA-v0`. No source code or base64 overlay is embedded in this notebook. Set `REPO_REF` in the config cell when you want to pin an exact branch or tag.
    """
)

notebook["cells"][3]["source"] = lines(
    """
    import json, os, random, shutil, subprocess, sys, time, zipfile
    from pathlib import Path

    WORK = Path("/kaggle/working")
    INPUT = Path("/kaggle/input")
    REPO = WORK / "RUPA-DSA-v0"
    ARTIFACTS = WORK / "rupa_v0_artifacts"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if REPO.exists():
        shutil.rmtree(REPO)
    clone_cmd = ["git", "clone", "--depth", "1"]
    if REPO_REF:
        clone_cmd += ["--branch", REPO_REF]
    clone_cmd += [REPO_URL, str(REPO)]
    subprocess.run(clone_cmd, check=True)
    os.chdir(REPO)

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "ftfy", "regex", "einops==0.8.0", "ipdb", "scikit-learn", "pandas"
    ], check=True)

    import numpy as np
    import pandas as pd
    import torch

    print("Repository:", REPO_URL, "ref:", REPO_REF or "default branch")
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    assert torch.cuda.is_available(), "Enable Accelerator = GPU in Kaggle Notebook Settings."
    assert "RUPA-DSA" in (REPO / "src/model.py").read_text(encoding="utf-8")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("Direct GitHub clone ready.")
    """
)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Updated {NOTEBOOK}")
