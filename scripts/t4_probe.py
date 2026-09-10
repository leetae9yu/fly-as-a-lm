# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3"]
# ///
# Run: colab exec -s flyrl-language-t4 -f scripts/t4_probe.py
"""Require a real T4 and execute a tensor operation on that device."""

import json
import platform
import sys

import torch

if not torch.cuda.is_available():
    message = "CUDA is unavailable; the language run requires a T4."
    raise RuntimeError(message)
device_name = torch.cuda.get_device_name(0)
if "T4" not in device_name:
    message = f"Expected a free-tier T4, received {device_name}."
    raise RuntimeError(message)
tensor = torch.ones((32, 32), device="cuda")
product = tensor @ tensor
torch.cuda.synchronize()
_ = sys.stdout.write(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": device_name,
            "vram_bytes": torch.cuda.mem_get_info()[1],
            "tensor_device": str(product.device),
            "matmul_value": product[0, 0].item(),
        }
    )
    + "\n"
)
