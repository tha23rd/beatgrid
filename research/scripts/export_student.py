"""Export a trained student (DATA/models/<name>.pt) to the safetensors file the Rust crate
embeds (crates/beatgrid/models/student.safetensors).

  uv run --group dev python scripts/export_student.py student-v2
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from beatgrid_lab.config import DATA, REPO

name = sys.argv[1]
checkpoint = torch.load(DATA / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
tensors = {k: v.contiguous().float() for k, v in checkpoint["state_dict"].items()}
out = REPO / "crates" / "beatgrid" / "models" / "student.safetensors"
save_file(tensors, str(out), metadata={"name": name, "config": json.dumps(checkpoint["config"]),
                                       "step": str(checkpoint.get("step")), "val": json.dumps(checkpoint.get("val"))})
print(f"{name} (step {checkpoint.get('step')}, {sum(v.numel() for v in tensors.values()) / 1e3:.0f}k params) -> {out} "
      f"({Path(out).stat().st_size / 1e3:.0f} kB)")
