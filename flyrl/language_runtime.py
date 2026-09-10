"""Explicit deterministic tensor placement and content identities."""

import hashlib
import json
import os

import numpy as np
import torch

from flyrl.connectome import Graph
from flyrl.language_models import DeviceEvidence, LanguageError, Ports


def configure_device(requested: str) -> torch.device:
    """Reject unavailable CUDA instead of changing the requested experiment."""
    if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
        raise LanguageError(reason="device must be cpu, cuda, or cuda:<index>")
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise LanguageError(reason=f"invalid device: {requested}") from error
    if device.type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise LanguageError(
                reason="CUDA requested but unavailable; no CPU fallback"
            )
        if requested.startswith("cuda:") and device.index >= torch.cuda.device_count():
            raise LanguageError(reason="CUDA device index is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)
    torch.set_num_threads(1)
    return device


def select_ports(nodes: int, symbols: int, seed: int) -> Ports:
    """Permute indices without observing edges or edge weights."""
    if nodes < 2 * symbols + 1:
        raise LanguageError(
            reason="need two neurons per symbol and recurrent remainder"
        )
    order: np.ndarray[tuple[int], np.dtype[np.int64]] = np.arange(nodes, dtype=np.int64)
    np.random.default_rng(np.random.SeedSequence([seed, 11])).shuffle(order)
    return Ports(
        tuple(order.item(i) for i in range(symbols)),
        tuple(order.item(i) for i in range(symbols, 2 * symbols)),
    )


def graph_fingerprint(graph: Graph) -> str:
    """Hash ordered identities, signed weights, and topology, excluding prose."""
    digest = hashlib.sha256(json.dumps(graph.node_ids).encode())
    for array in (graph.source, graph.target, graph.weight):
        digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash exact contiguous float32 bytes on the host."""
    array = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def device_evidence(tensor: torch.Tensor, requested: str) -> DeviceEvidence:
    """Read actual tensor placement and current CUDA allocation counters."""
    cuda = tensor.device.type == "cuda"
    return DeviceEvidence(
        requested=requested,
        tensor_device=str(tensor.device),
        name=torch.cuda.get_device_name(tensor.device) if cuda else "CPU",
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        allocated_bytes=torch.cuda.memory_allocated(tensor.device) if cuda else 0,
        peak_allocated_bytes=(
            torch.cuda.max_memory_allocated(tensor.device) if cuda else 0
        ),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
