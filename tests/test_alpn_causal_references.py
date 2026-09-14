"""Compact CPU-only frozen-head packaging checks."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from pydantic import ValidationError

import scripts.alpn_causal_references as refs
from flyrl.regional_probe_types import ExtractionConfig, ProbeConfig
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import CompactProbeResult, HeadArtifact
from scripts.regional_probe_artifacts import write_npz
from scripts.regional_probe_manifest import (
    ProbeGroupRecord,
    RegionalGroupManifest,
    SeedGroupPlan,
)
from scripts.regional_probe_protocol import ProbeSourceCheckpoint, RegionalProbeProtocol


@dataclass(frozen=True, slots=True)
class Fixture:
    regional: Path
    groups: Path
    files: tuple[tuple[str, bytes], ...]


@pytest.fixture
def head_fault() -> str:
    return ""


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head_fault: str) -> Fixture:
    groups = tuple(
        ProbeGroupRecord(
            name=name,
            draw=draw,
            indices=tuple(range(97)),
            node_ids=tuple(str(i) for i in range(97)),
        )
        for name in ("alpn", "trained_readout")
        for draw in range(5 if name == "alpn" else 1)
    )
    plans = tuple(
        SeedGroupPlan(
            seed=seed,
            sensory_indices=(100,),
            trained_readout_indices=tuple(range(97)),
            groups=groups,
        )
        for seed in range(7, 13)
    )
    group_manifest = RegionalGroupManifest(
        graph_sha256="a" * 64,
        port_manifest_sha256="b" * 64,
        port_arrays_sha256="c" * 64,
        selection_seed=1,
        group_size=97,
        draws=5,
        seeds=plans,
    )
    group_path = tmp_path / "groups.json"
    _ = group_path.write_text(group_manifest.model_dump_json())
    sources = tuple(
        ProbeSourceCheckpoint.model_validate(
            {
                **dict.fromkeys(ProbeSourceCheckpoint.model_fields, "a" * 64),
                "seed": seed,
                "wiring": wiring,
                "archive_bytes": 1,
                "filename": f"seed-{seed}-{wiring}-random_random.zip",
            }
        )
        for seed in range(7, 13)
        for wiring in ("real", "shuffled")
    )
    protocol = RegionalProbeProtocol.model_validate(
        {
            **dict.fromkeys(
                RegionalProbeProtocol.model_fields.keys() - {"format"}, "unused"
            ),
            "graph_nodes": 100,
            "graph_edges": 1,
            "group_size": 97,
            "draws": 5,
            "group_manifest_sha256": file_digest(group_path),
            "checkpoints": sources,
            "extraction": ExtractionConfig(),
            "probe": ProbeConfig(vocab_size=2, updates=0),
            "heads_per_checkpoint": 6,
            "practical_threshold": 0.1,
        }
    )
    stats = {
        "feature_sha256": "a" * 64,
        "label_sha256": "b" * 64,
        "variance": [0.0] * 97,
        "saturation_fraction": 0.0,
    }
    metric = {"tokens": 1, "nll": 1.0, "perplexity": None, "accuracy": 0.0}
    arrays_path = tmp_path / "head.npz"
    write_npz(
        arrays_path,
        {
            "weight": np.zeros(
                (96 if head_fault == "shape" else 97, 2), dtype=np.float32
            ),
            "bias": np.zeros(2, dtype=np.float32),
            "mean": np.zeros(97, dtype=np.float32),
            "std": np.full(97, 0 if head_fault == "floor" else 1, dtype=np.float32),
        },
    )
    files: list[tuple[str, bytes]] = []
    regional = tmp_path / "regional.zip"
    with ZipFile(regional, "w") as outer:
        outer.writestr("protocol.json", protocol.model_dump_json())
        outer.writestr("summary.json", b"MUST NOT READ")
        for source in sources:
            buffer = BytesIO()
            with ZipFile(buffer, "w") as inner:
                for name in ("source.json", "heldout.npz"):
                    inner.writestr(name, b"MUST NOT READ")
                for group in groups:
                    stem = f"{group.name}-{group.draw}"
                    artifact = HeadArtifact(
                        source=source,
                        group=group,
                        extraction=protocol.extraction,
                        result=CompactProbeResult.model_validate(
                            {
                                "config": protocol.probe.model_copy(
                                    update={"seed": source.seed}
                                ),
                                "parameter_count": 196,
                                "trace": (),
                                "features": (stats, stats, stats),
                                "valid": metric,
                                "test": metric,
                            }
                        ),
                        parameters_file=f"{stem}.npz",
                        parameters_sha256=file_digest(arrays_path),
                    )
                    for ext, original in (
                        ("json", artifact.model_dump_json().encode()),
                        ("npz", arrays_path.read_bytes()),
                    ):
                        name = f"{stem}.{ext}"
                        payload = original
                        replacements = {
                            "source": (b'"seed":7', b'"seed":6'),
                            "group": (b'"node_ids":["0"', b'"node_ids":["999"'),
                            "config": (b'"batch_size":256', b'"batch_size":257'),
                            "extraction": (b'"chunk_size":128', b'"chunk_size":127'),
                            "hash": (
                                b'"parameters_sha256":"',
                                b'"parameters_sha256":"x',
                            ),
                        }
                        if ext == "json" and head_fault in replacements:
                            payload = payload.replace(*replacements[head_fault], 1)
                        inner.writestr(name, payload)
                        files.append(
                            (f"seed-{source.seed}-{source.wiring}/{name}", payload)
                        )
            outer.writestr(
                f"seed-{source.seed}-{source.wiring}-probes.zip", buffer.getvalue()
            )
    digest = sha256()
    for name, payload in files:
        digest.update(name.encode())
        digest.update(sha256(payload).digest())
    monkeypatch.setattr(refs, "RESULT_SHA256", file_digest(regional))
    monkeypatch.setattr(
        refs, "PROTOCOL_SHA256", sha256(protocol.model_dump_json().encode()).hexdigest()
    )
    monkeypatch.setattr(refs, "ORDERED_SHA256", digest.hexdigest())
    monkeypatch.setattr(refs, "TOTAL_BYTES", sum(len(payload) for _, payload in files))
    if head_fault in ("result", "protocol"):
        monkeypatch.setattr(refs, f"{head_fault.upper()}_SHA256", "0" * 64)
    if head_fault == "groups":
        _ = group_path.write_bytes(group_path.read_bytes() + b" ")
    return Fixture(regional, group_path, tuple(files))


def test_roundtrip_byte_identity_and_determinism(
    tmp_path: Path, frozen: Fixture
) -> None:
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    manifest = refs.package_references(frozen.regional, frozen.groups, first)
    assert refs.package_references(frozen.regional, frozen.groups, second) == manifest
    restored = refs.ReferenceManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    with pytest.raises(ValidationError):
        manifest.result_sha256 = "0" * 64
    for update in ({"unknown": 1}, {"files": manifest.files[:-1]}):
        with pytest.raises(ValidationError):
            _ = refs.ReferenceManifest.model_validate(manifest.model_dump() | update)
    record = manifest.files[0].model_dump()
    for update in ({"size": "1"}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            _ = refs.ReferenceFile.model_validate(record | update)
    with ZipFile(first) as archive:
        for name, payload in frozen.files:
            assert archive.read(name.replace("/", "-")) == payload
    with pytest.raises(FileExistsError):
        _ = refs.package_references(frozen.regional, frozen.groups, first)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "change", ["extra", "missing", "nested", "duplicate", "bytes", "order"]
)
def test_archive_rejects_corruption(
    tmp_path: Path, frozen: Fixture, change: str
) -> None:
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    manifest = refs.package_references(frozen.regional, frozen.groups, good)
    entries = [(name.replace("/", "-"), data) for name, data in frozen.files]
    entries = {
        "extra": [*entries, ("extra", b"x")],
        "missing": entries[:-1],
        "nested": [("nested/" + entries[0][0], entries[0][1]), *entries[1:]],
        "duplicate": [*entries[:-1], entries[0]],
        "bytes": [(entries[0][0], entries[0][1] + b" "), *entries[1:]],
        "order": list(reversed(entries)),
    }[change]
    with ZipFile(bad, "w") as archive:
        for name, data in entries:
            if change == "duplicate" and name in archive.namelist():
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, data)
            else:
                archive.writestr(name, data)
    with pytest.raises(ValueError, match="Reference archive"):
        refs.validate_reference_archive(bad, manifest)


@pytest.mark.parametrize(
    "head_fault",
    [
        "source",
        "group",
        "config",
        "extraction",
        "hash",
        "shape",
        "floor",
        "result",
        "protocol",
        "groups",
    ],
)
def test_rejects_invalid_heads(tmp_path: Path, frozen: Fixture) -> None:
    with pytest.raises(ValueError, match=r"identity|validation"):
        _ = refs.package_references(
            frozen.regional, frozen.groups, tmp_path / "bad.zip"
        )
    assert not (tmp_path / "bad.zip").exists()
