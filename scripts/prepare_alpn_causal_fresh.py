# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2"]
# ///
"""Materialize the frozen ALPN causal fresh-evaluation corpus artifact."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.alpn_causal_fresh import (
    build_fresh_artifact,
    validate_saved_artifact,
)
from scripts.alpn_causal_fresh_archive import save_fresh_artifact

if TYPE_CHECKING:
    from scripts.alpn_causal_fresh_types import FreshArtifact


def prepare(root: Path) -> FreshArtifact:
    """Build, save, and independently validate the exact fresh-evaluation artifact."""
    artifact = build_fresh_artifact(root)
    save_fresh_artifact(artifact, root)
    return validate_saved_artifact(root)


def main() -> None:
    """Generate the artifact below the repository containing this script."""
    artifact = prepare(Path(__file__).resolve().parents[1])
    _ = sys.stdout.write(
        " ".join(
            (
                f"stories={artifact.story_count}",
                f"targets={artifact.target_count}",
                f"token_sha256={artifact.token_sha256}",
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
