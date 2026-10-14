"""Merge an I2 LoRA into Qwen3.5-2B and export GGUF for the CPU scorer (semif.modelscorer).

    python -m ix.export --name i2_r3          # runs/i2_r3/merged -> runs/i2_r3/gguf/{f16,Q8_0,Q4_K_M}.gguf

Needs llama.cpp's convert_hf_to_gguf.py and llama-quantize (set LLAMA_CPP_CONVERT / LLAMA_CPP_BIN if they are not
in the Homebrew locations below). The merge runs on CPU, so it works while the GPU is busy.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from . import BASE_MODEL, RUNS

LLAMA_BIN = Path(os.environ.get("LLAMA_CPP_BIN", "/home/linuxbrew/.linuxbrew/bin"))
CONVERT = Path(os.environ.get("LLAMA_CPP_CONVERT",
                              "/home/linuxbrew/.linuxbrew/Cellar/llama.cpp/8960/bin/convert_hf_to_gguf.py"))
# The converter fingerprints the tokenizer by hashing an encoding; this transformers version gives a hash for the
# Qwen3.5 tokenizer that the converter does not list, although tokenizer.json is identical to a listed one.
LOCAL_HASH = "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"


def converter() -> Path:
    patched = RUNS / "convert_hf_to_gguf_patched.py"
    src = CONVERT.read_text()
    anchor = '        if chkhsh == "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4":'
    assert anchor in src, "converter changed; update the patch anchor"
    patched.write_text(src.replace(anchor, f'        if chkhsh == "{LOCAL_HASH}":\n            res = "qwen35"\n' + anchor))
    return patched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--quants", default="Q8_0,Q4_K_M")
    args = ap.parse_args()
    run = RUNS / args.name
    merged, out = run / "merged", run / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    if not (merged / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(model, run / "adapter").merge_and_unload()
        model.save_pretrained(merged)
        from huggingface_hub import snapshot_download   # original tokenizer files keep the converter's fingerprint
        snap = Path(snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.txt", "*.jinja"]))
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
            if (snap / f).exists():
                shutil.copy(snap / f, merged / f)
        print("merged ->", merged, flush=True)
    f16 = out / "f16.gguf"
    if not f16.exists():
        subprocess.run([sys.executable, str(converter()), str(merged), "--outtype", "f16", "--outfile", str(f16)],
                       check=True)
    for q in args.quants.split(","):
        dst = out / f"{q}.gguf"
        if not dst.exists():
            subprocess.run([str(LLAMA_BIN / "llama-quantize"), str(f16), str(dst), q], check=True)
        print(q, f"{dst.stat().st_size / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
