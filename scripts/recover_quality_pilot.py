"""Strict ZIP orchestration for portable quality recovery, never training or resume."""

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Annotated, Final
from zipfile import BadZipFile, ZipFile

import torch
import typer

from flyrl.ar_learning import ARLearner
from flyrl.connectome import load_graph
from flyrl.language_runtime import graph_fingerprint
from flyrl.quality_pilot_schema import QualityProgress, QualityProtocol, select_update
from flyrl.quality_recovery import RecoveryReport, verify
from flyrl.quality_recovery_validation import figures, read_model, require
from flyrl.story_data import StoryCorpus, load_story_corpus
from scripts.connectome_source import file_digest

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def load_inputs(protocol: QualityProtocol) -> tuple[ARLearner, StoryCorpus]:
    """Check trusted prepared inputs without consulting ignored raw story sources."""
    for artifact in (protocol.graph, protocol.corpus):
        require(
            file_digest(artifact.path) == artifact.sha256,
            "Protocol input hash mismatch",
        )
    graph, stories = (
        load_graph(protocol.graph.path),
        load_story_corpus(protocol.corpus.path),
    )
    require(
        graph_fingerprint(graph) == protocol.graph.fingerprint
        and (len(graph.node_ids), graph.source.size)
        == (protocol.graph.nodes, protocol.graph.edges),
        "Protocol graph identity/count mismatch",
    )
    require(
        stories.corpus.fingerprint == protocol.corpus.fingerprint
        and len(stories.corpus.vocabulary) == protocol.corpus.vocabulary
        and tuple(len(split.sha256) for split in stories.metadata.splits)
        == protocol.corpus.stories
        and tuple(int(tokens.size) for tokens in stories.streams)
        == protocol.corpus.tokens,
        "Protocol corpus identity/count mismatch",
    )
    require(
        all(stories.corpus.encode(prompt) for prompt in protocol.prompts),
        "Protocol prompt identity mismatch",
    )
    _ = stories.starts(stories.metadata.train, protocol.config.context)
    torch.set_num_threads(protocol.cpu_threads)
    return ARLearner(
        graph, protocol.config.model_copy(update={"device": "cpu"})
    ), stories


def activation_paths(
    protocol: QualityProtocol, update: int, selected: int
) -> tuple[str, ...]:
    """Derive recording locations from the trusted protocol, not the manifest."""
    return (
        tuple(
            f"activations-{update:06d}/prompt-{index:02d}/activations.json"
            for index in protocol.activation_prompt_indices
        )
        if update in {protocol.early_update, selected}
        else ()
    )


def members(archive: ZipFile, expected: set[str]) -> None:
    """Validate names, Unix types, exact membership and CRCs before extraction."""
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate archive members")
    parents = {
        parent.as_posix() + "/"
        for name in expected
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    files: set[str] = set()
    for item in archive.infolist():
        name = item.filename.removesuffix("/") if item.is_dir() else item.filename
        require(
            bool(name)
            and item.orig_filename == item.filename
            and not name.startswith("/")
            and not any(part in {"", ".", ".."} for part in name.split("/"))
            and not any(char in name for char in ("\\", ":", "\x00")),
            "Unsafe archive member path",
        )
        kind = stat.S_IFMT(item.external_attr >> 16)
        require(
            kind in ({0, stat.S_IFDIR} if item.is_dir() else {0, stat.S_IFREG}),
            "Nonregular archive member",
        )
        if item.is_dir():
            require(item.filename in parents, "Extra archive directory")
        else:
            files.add(item.filename)
    require(files == expected, "Archive extras or missing files")
    require(archive.testzip() is None, "Archive CRC mismatch")


def recover(archive: Path, protocol: Path, destination: Path) -> RecoveryReport:
    """Validate ZIP before staging; publish only after every semantic gate succeeds."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    trusted = read_model(QualityProtocol, protocol)
    learner, stories = load_inputs(trusted)
    try:
        with ZipFile(archive) as zipped:
            report = QualityProgress.model_validate_json(
                zipped.read("report.json"), strict=True
            )
            require(
                report
                == QualityProgress.model_validate_json(
                    zipped.read("progress.json"), strict=True
                ),
                "Torn progress/report",
            )
            require(
                report.protocol == trusted
                and report.protocol_sha256 == trusted.sha256
                and QualityProtocol.model_validate_json(
                    zipped.read("protocol.json"), strict=True
                )
                == trusted,
                "Protocol identity mismatch",
            )
            require(
                report.updates == trusted.schedule.target_updates
                and report.next_lr is None
                and tuple(v.update for v in report.validation)
                == trusted.evaluation_updates,
                "Incomplete target/evaluation milestones",
            )
            selected = select_update(report.validation)
            require(
                report.selected_update == selected
                and tuple(r.update for r in report.completed)
                == tuple(
                    sorted(
                        {
                            trusted.early_update,
                            selected,
                            trusted.schedule.target_updates,
                        }
                    )
                ),
                "Selected checkpoint/test milestones mismatch",
            )
            expected = {
                "protocol.json",
                "progress.json",
                "report.json",
                "checkpoint-latest.npz",
            }
            expected.update(
                f"{kind}-{u:06d}.{ext}"
                for u in trusted.evaluation_updates
                for kind, ext in (("checkpoint", "npz"), ("evaluation", "json"))
            )
            for result in report.completed:
                expected.add(f"result-{result.update:06d}.json")
                paths = activation_paths(trusted, result.update, selected)
                require(
                    result.activations == paths,
                    "Activation recording membership mismatch",
                )
                artifacts = tuple(
                    (PurePosixPath(path).parent / name).as_posix()
                    for path in paths
                    for name in sorted(
                        (
                            "activations.json",
                            "activations.npz",
                            *figures(trusted.generation.length),
                        )
                    )
                )
                require(
                    tuple(a.path for a in result.artifacts) == artifacts,
                    "Activation artifact membership mismatch",
                )
                expected.update(artifacts)
            members(zipped, expected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                prefix=".quality-recovery-", dir=destination.parent
            ) as temporary:
                root = Path(temporary) / "output"
                root.mkdir()
                for name in sorted(expected):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(name) as source, path.open("xb") as target:
                        while chunk := source.read(1024 * 1024):
                            _ = target.write(chunk)
                require(
                    file_digest(root / "checkpoint-latest.npz")
                    == report.checkpoint_sha256
                    == report.validation[-1].checkpoint_sha256,
                    "Latest checkpoint/progress hash mismatch",
                )
                recovered = verify(root, report, learner, stories)
                _ = (root / "recovery.json").write_text(
                    recovered.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                _ = root.rename(destination)
    except (BadZipFile, KeyError, RuntimeError, EOFError, OSError) as error:
        message = f"Invalid quality archive: {error}"
        raise ValueError(message) from error
    typer.echo("QUALITY_RECOVERY_VERIFIED")
    return recovered


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Recover one complete result ZIP against a separately trusted protocol."""

    archive: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    protocol: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    destination: Annotated[Path, typer.Option(file_okay=False)]

    def __post_init__(self) -> None:
        """Execute strict recovery without hyperparameter or identity overrides."""
        _ = recover(self.archive, self.protocol, self.destination)


if __name__ == "__main__":
    app()
