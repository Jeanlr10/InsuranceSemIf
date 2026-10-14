"""Project IX training code: I2 is Qwen3.5-2B with a LoRA trained on the option-letter readout, with GLM 5.3 Flash
as an adversarial attacker in the loop and an invariance loss between clean and injected inputs.

Needs torch, transformers and peft (requirements-train.txt). The semif package itself stays standard-library only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ix"          # generated training and test data (published, MIT)
RUNS = ROOT / "runs"                 # checkpoints and scores (not published)
BASE_MODEL = "Qwen/Qwen3.5-2B"
