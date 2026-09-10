# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run after ar_bootstrap.py on Colab:
# colab exec -s flyrl-autoregressive-t4 -f scripts/ar_profile.py
"""Identify GPU kernel costs for a single full-connectome forward/backward step."""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import load_graph


@runtime_checkable
class ProfileTable(Protocol):
    """Typed boundary around the profiler's unannotated rendering method."""

    def table(self, *, sort_by: str, row_limit: int) -> str:
        """Render the measured operation table."""
        ...


root = Path("/content/flyrl-ar")
graph = load_graph(root / "data" / "large_connectome" / "malecns_v1_full.npz")
learner = ARLearner(
    graph, ARConfig(alphabet_size=48, device="cuda", batch_size=8, context=1)
)
window = torch.arange(16, device="cuda").reshape(8, 2)
torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as profile:
    loss = learner.loss(window)
    torch.autograd.backward(loss)
    torch.cuda.synchronize()
events = TypeAdapter(
    ProfileTable, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(profile.key_averages())
table = events.table(sort_by="self_device_time_total", row_limit=20)
_ = (root / "results" / "ar-capacity" / "kernel-profile.txt").write_text(table)
_ = sys.stdout.write(table + "\nAR_PROFILE_COMPLETE\n")
del learner, graph, loss, window
torch.cuda.empty_cache()
