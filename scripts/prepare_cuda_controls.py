"""Fetch and verify the exact public inputs for the separate CUDA study."""
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parm.data.download import fetch, sha256_of
from huggingface_hub import HfApi, snapshot_download


def main():
    if not fetch("MOSI", ROOT / "datasets"):
        raise RuntimeError("Dataset hash verification failed")
    revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if not offline and HfApi().model_info("bert-base-uncased", revision="main").sha != revision:
        raise ValueError("BERT main revision changed; explicit cache resolution required")
    snapshot = Path(snapshot_download(
        "bert-base-uncased", revision="main", local_files_only=offline,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "vocab.txt"]))
    if snapshot.name != revision:
        raise ValueError("Unexpected BERT snapshot revision")
    digest = sha256_of(snapshot / "model.safetensors")
    expected = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
    if digest != expected:
        raise ValueError("BERT weights differ from the local control inputs")
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements-controls-cuda-resolved.txt").write_text(packages)
    result = dict(dataset_sha256=sha256_of(ROOT / "datasets/MOSI/unaligned_50.pkl"),
                  bert_revision=snapshot.name, bert_sha256=digest,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),
                  attention=model.config._attn_implementation,
                  precision="bf16", conditions=40, concurrent_jobs=4)
    (ROOT / "cuda_input_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
