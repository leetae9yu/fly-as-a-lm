# /// script
# requires-python = ">=3.11"
# dependencies = ["flyrl[language]", "typer>=0.15,<1"]
# ///
# Run from the repository: uv run python -m scripts.recover_babi_task1
"""Read-only local verification of a completed, separately sealed Task 1 bundle."""

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final

import typer

from flyrl.babi_data import BabiError, load_tokenizer
from flyrl.babi_protocol import BabiProtocol, code_identity
from flyrl.babi_recovery import Inputs, verify
from flyrl.babi_recovery_validation import read_model, require
from flyrl.babi_storage import (
    MEMBERS,
    Metadata,
    Split,
    TrainValid,
    load_test,
    load_train_valid,
)
from flyrl.connectome import load_graph
from flyrl.language_runtime import graph_fingerprint
from scripts.connectome_source import file_digest

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def regular(path: Path) -> None:
    """Reject links in any path component and nonregular artifact leaves."""
    require(
        not any(p.is_symlink() for p in (path, *path.parents)), "Unsafe symlink path"
    )
    require(stat.S_ISREG(path.stat().st_mode), "Artifact must be regular")


def members(root: Path, expected: set[str]) -> None:
    """Check exact files and implied directories, without following symlink aliases."""
    require(
        root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
        "Unsafe bundle root",
    )
    for name in expected:
        require(
            bool(name)
            and not name.startswith("/")
            and not any(p in {"", ".", ".."} for p in name.split("/"))
            and not any(c in name for c in ("\\", ":", "\x00")),
            "Unsafe artifact path",
        )
    directories = {
        str(parent)
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    paths = tuple(root.rglob("*"))
    require(all(not p.is_symlink() for p in paths), "Unsafe bundle symlink")
    actual_files = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    actual_dirs = {p.relative_to(root).as_posix() for p in paths if p.is_dir()}
    require(
        actual_files == expected
        and actual_dirs == directories
        and len(paths) == len(expected) + len(directories),
        "Bundle membership mismatch",
    )
    for name in expected:
        regular(root / name)


def load_inputs(protocol: BabiProtocol) -> Inputs:
    """Authenticate local source/graph/code identities without creating a learner."""
    regular(protocol.graph.path)
    require(
        file_digest(protocol.graph.path) == protocol.graph.sha256
        and protocol.code == code_identity(),
        "Protocol graph/code hash mismatch",
    )
    for entry in protocol.code:
        regular(entry.path)
    graph = load_graph(protocol.graph.path)
    require(
        graph_fingerprint(graph) == protocol.graph.fingerprint
        and (len(graph.node_ids), graph.source.size)
        == (protocol.graph.nodes, protocol.graph.edges),
        "Protocol graph identity mismatch",
    )
    source = protocol.corpus
    suffix = "npz" if protocol.profile == "quality" else "json"
    for name, digest in (
        (f"train_valid.{suffix}", source.train_valid_sha256),
        (f"test.{suffix}", source.test_sha256),
        ("tokenizer.json", source.tokenizer_sha256),
    ):
        regular(source.path / name)
        require(file_digest(source.path / name) == digest, "Input bundle hash mismatch")
    if protocol.profile == "quality":
        regular(source.path / "provenance.json")
        regular(source.path / "SOURCE_LICENSE.txt")
        metadata = read_model(Metadata, source.path / "provenance.json")
        require(
            metadata.train_valid_sha256 == source.train_valid_sha256
            and metadata.test_sha256 == source.test_sha256
            and metadata.split_fingerprint == source.train_valid_fingerprint
            and metadata.provenance == source.provenance
            and (source.path / "SOURCE_LICENSE.txt").stat().st_size == MEMBERS[2].size,
            "Official source/license identity mismatch",
        )
        corpus, test = load_train_valid(source.path), load_test(source.path)
    else:
        corpus, test = (
            read_model(TrainValid, source.path / "train_valid.json"),
            read_model(Split, source.path / "test.json"),
        )
    require(
        corpus.fingerprint == source.train_valid_fingerprint
        and (
            len(corpus.train.questions),
            len(corpus.valid.questions),
            len(test.questions),
        )
        == source.counts,
        "Input split identity mismatch",
    )
    return Inputs(corpus, test, load_tokenizer(source.path / "tokenizer.json"))


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Verify one completed bundle and exclusively publish recomputed evidence."""

    protocol: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    root: Annotated[Path, typer.Option(exists=True, file_okay=False)]
    output: Annotated[Path, typer.Option(dir_okay=False)]

    def __post_init__(self) -> None:
        """Run the strict read-only checks before creating the separate report."""
        if self.output.exists():
            raise FileExistsError(self.output)
        if self.output.resolve().is_relative_to(self.root.resolve()):
            raise BabiError(reason="Recovery report must remain outside the bundle")
        frozen = BabiProtocol.model_validate_json(self.protocol.read_bytes())
        result = verify(self.root, frozen, load_inputs(frozen))
        with self.output.open("x", encoding="utf-8") as stream:
            _ = stream.write(result.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        typer.echo("BABI_TASK1_RECOVERY_VERIFIED")


if __name__ == "__main__":
    app()
