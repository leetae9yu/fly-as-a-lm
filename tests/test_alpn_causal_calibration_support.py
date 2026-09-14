"""Train-only file access and the independently trusted runtime/code boundary."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from flyrl.story_data import StoryMetadata, StorySplit
from scripts import alpn_causal_calibration_worker as worker
from scripts.alpn_causal_calibration_support import (
    check_files,
    load_context,
    old_training,
)
from scripts.alpn_causal_calibration_types import FileIdentity
from scripts.alpn_causal_calibration_worker import restore_checkpoint
from scripts.connectome_source import file_digest
from scripts.regional_probe_protocol import RegionalProbeProtocol
from tests.test_alpn_causal_calibration_worker import restoration, synthetic_context

TOKENIZER = "synthetic tokenizer"
TOKENIZER_HASH = sha256(TOKENIZER.encode()).hexdigest()


def training_corpus(root: Path) -> RegionalProbeProtocol:
    context, _ = synthetic_context(root)
    empty = StorySplit(offsets=(0,), sha256=())
    metadata = StoryMetadata(
        source="synthetic",
        revision="synthetic",
        license_text="synthetic",
        consumed=(),
        duplicates_removed=0,
        seed=7,
        requested_vocab_size=4096,
        tokenizers_version="0.22.2",
        train=context.split,
        valid=empty,
        test=empty,
    )
    path = root / "corpus.npz"
    np.savez(
        path,
        train=context.tokens,
        tokenizer_json=np.asarray(TOKENIZER),
        fingerprint=np.asarray(context.inputs.protocol.corpus_fingerprint),
        provenance=np.asarray(metadata.model_dump_json()),
    )
    with ZipFile(path, "a") as archive:
        for name in ("valid.npy", "test.npy"):
            archive.writestr(name, b"INVALID NPY: MUST NEVER BE MATERIALIZED")
    _ = (root / "alpn_causal_fresh.npz").write_bytes(b"MUST NEVER BE OPENED")
    return context.inputs.protocol.model_copy(
        update={"corpus_sha256": file_digest(path)}
    )


def test_training_loader_never_materializes_heldout_or_fresh_arrays(
    tmp_path: Path,
) -> None:
    protocol = training_corpus(tmp_path)
    tokens, split = old_training(tmp_path, protocol, tokenizer_sha256=TOKENIZER_HASH)
    assert tokens.dtype == np.int64
    assert tokens.size - len(split.sha256) == 229745
    assert not tokens.flags.writeable
    with ZipFile(tmp_path / "corpus.npz") as archive:
        assert archive.read("valid.npy").startswith(b"INVALID NPY")


def test_corpus_hash_is_checked_before_parsing(tmp_path: Path) -> None:
    protocol = training_corpus(tmp_path)
    _ = (tmp_path / "corpus.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="corpus hash"):
        _ = old_training(tmp_path, protocol, tokenizer_sha256=TOKENIZER_HASH)


def test_duplicate_corpus_member_is_not_silently_selected(tmp_path: Path) -> None:
    protocol = training_corpus(tmp_path)
    path = tmp_path / "corpus.npz"
    with (
        ZipFile(path, "a") as archive,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        archive.writestr("train.npy", b"duplicate")
    protocol = protocol.model_copy(update={"corpus_sha256": file_digest(path)})
    with pytest.raises(ValueError, match="Duplicate"):
        _ = old_training(tmp_path, protocol, tokenizer_sha256=TOKENIZER_HASH)


@pytest.mark.parametrize("fault", ["duplicate", "traversal", "absolute"])
def test_input_seal_rejects_duplicate_or_escaping_paths(
    tmp_path: Path, fault: str
) -> None:
    context, _ = synthetic_context(tmp_path)
    original = context.seal.files[0]
    files = (original, original)
    if fault != "duplicate":
        files = (
            original.model_copy(
                update={"path": "../escape" if fault == "traversal" else "/escape"}
            ),
        )
    with pytest.raises(ValueError, match=r"Duplicate|path"):
        check_files(tmp_path, context.seal.model_copy(update={"files": files}))


def test_sealed_project_code_must_match_the_executing_code(tmp_path: Path) -> None:
    context, _ = synthetic_context(tmp_path)
    name = "scripts/alpn_causal_calibration_worker.py"
    path = tmp_path / name
    path.parent.mkdir()
    _ = path.write_text("changed implementation\n")
    identity = FileIdentity(
        path=name, sha256=file_digest(path), size=path.stat().st_size
    )
    seal = context.seal.model_copy(update={"files": (identity,)})
    with pytest.raises(ValueError, match="hash"):
        _ = load_context(tmp_path, seal)


def test_real_restoration_rejects_cpu_before_parsing_checkpoint(tmp_path: Path) -> None:
    context, _ = synthetic_context(tmp_path)
    with pytest.raises(ValueError, match="Tesla T4"):
        _ = restore_checkpoint(context, context.inputs.protocol.checkpoints[0])


def test_balanced_requires_every_promised_coordinate_in_all_six_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, models = synthetic_context(tmp_path, balanced=True)
    monkeypatch.setattr(worker, "restore_checkpoint", restoration(models))
    result = worker.run_calibration(context, tmp_path / "balanced.zip")
    assert result.status == "balanced"
    assert all(
        balance.smd == 0.0 and balance.ks == 0.0
        for seed in result.seeds
        for control in seed.matching.controls
        for balance in control.balance
    )
    assert (
        capsys.readouterr().out.splitlines()[-1]
        == "ALPN_CALIBRATION_STATUS status=balanced"
    )


def test_default_tokenizer_identity_cannot_be_relaxed(tmp_path: Path) -> None:
    protocol = training_corpus(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        _ = old_training(tmp_path, protocol)
