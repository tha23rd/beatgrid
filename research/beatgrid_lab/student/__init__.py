"""Distilling a small, fast activation network from Beat This!.

  prepare.py  decode audio, compute the log-mel input and the teacher's beat/downbeat
              probabilities (GPU), one .npz per track
  train.py    fit the student to the teacher (and to labels where a reference set has them)
  model.py    the student network

Runs on a CUDA machine; see research/scripts/setup-gpu-env.sh.
"""
