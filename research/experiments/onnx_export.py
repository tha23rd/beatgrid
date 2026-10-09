"""Export Beat This! to ONNX at a fixed chunk size and compare speed and output with PyTorch.

  uv run --group perf python experiments/onnx_export.py [--checkpoint final0] [--out DIR]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from beatgrid_lab.config import DATA

CHUNK = 1500  # frames: Beat This! runs 30-second chunks at 50 fps
MELS = 128


class Wrapped(torch.nn.Module):
    """(batch, frames, mels) log-mel -> (batch, frames, 2) beat/downbeat probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spect):
        out = self.model(spect)
        return torch.sigmoid(torch.stack([out["beat"], out["downbeat"]], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="final0")
    ap.add_argument("--out", default=str(DATA / "models"))
    args = ap.parse_args()
    from beat_this.inference import load_model

    model = Wrapped(load_model(args.checkpoint, "cpu")).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"beat_this-{args.checkpoint}.onnx"
    example = torch.randn(1, CHUNK, MELS)
    t = time.perf_counter()
    torch.onnx.export(model, (example,), str(path), input_names=["spect"], output_names=["probs"],
                      dynamic_axes={"spect": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t:.1f}s")

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    qpath = out / f"beat_this-{args.checkpoint}.int8.onnx"
    quantize_dynamic(str(path), str(qpath), weight_type=QuantType.QInt8)
    print(f"quantized {qpath} ({qpath.stat().st_size / 1e6:.1f} MB)")

    x = torch.randn(8, CHUNK, MELS)  # eight chunks = 4 minutes of audio
    with torch.inference_mode():
        t = time.perf_counter()
        ref = torch.cat([model(x[i : i + 1]) for i in range(len(x))]).numpy()
        torch_s = time.perf_counter() - t
    print(f"pytorch, 8 chunks one at a time: {torch_s:.2f}s")
    for p in (path, qpath):
        for threads in (8, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            sess.run(None, {"spect": x[:1].numpy()})  # warm up
            for batch in (1, 8):
                t = time.perf_counter()
                got = np.concatenate([sess.run(None, {"spect": x[i : i + batch].numpy()})[0] for i in range(0, len(x), batch)])
                s = time.perf_counter() - t
                err = np.abs(got - ref).max()
                print(f"onnxruntime {p.name:32s} threads {threads} batch {batch}: {s:.2f}s  (max |Δprob| {err:.4f})")


if __name__ == "__main__":
    main()
