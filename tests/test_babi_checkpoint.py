"""Pickle-free transaction boundaries and exact CPU Adam/sample continuation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.babi_checkpoint import Checkpoints
from flyrl.babi_data import BabiError
from flyrl.babi_learning import BabiExample, BabiLearner
from flyrl.babi_pilot_schema import Validation
from flyrl.babi_protocol import (
    BabiProtocol,
    CorpusIdentity,
    code_identity,
    runtime_identity,
)
from flyrl.connectome import Graph, save_graph
from flyrl.language_runtime import graph_fingerprint
from flyrl.quality_pilot_schema import ArtifactIdentity
from scripts.connectome_source import file_digest


def fixture(
    tmp_path: Path,
) -> tuple[BabiProtocol, BabiLearner, tuple[BabiExample, ...]]:
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, target = np.nonzero(np.ones((6, 6)) - np.eye(6))
    graph = Graph(
        tuple(str(i) for i in range(6)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size),
        "synthetic",
    )
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    config = ARConfig(
        alphabet_size=4096,
        tokenization="bpe",
        trainable_codes=True,
        context=8,
        batch_size=2,
        readout_neurons=3,
    )
    protocol = BabiProtocol(
        profile="smoke",
        graph=ArtifactIdentity(
            path=path,
            sha256=file_digest(path),
            fingerprint=graph_fingerprint(graph),
            nodes=6,
            edges=30,
        ),
        corpus=CorpusIdentity(
            path=tmp_path,
            train_valid_sha256="a" * 64,
            test_sha256="b" * 64,
            train_valid_fingerprint="c" * 64,
            counts=(2, 2, 2),
        ),
        config=config,
        runtime=runtime_identity("cpu"),
        code=code_identity(),
        updates=4,
        milestones=(0, 2, 4),
        checkpoint_interval=1,
        recovery_interval=2,
    )
    examples = (
        BabiExample(example_id="a", prompt_ids=(10, 11), answer_ids=(2811, 199)),
        BabiExample(example_id="b", prompt_ids=(12,), answer_ids=(3885, 453, 199)),
    )
    return protocol, BabiLearner(graph, config), examples


def test_resume_when_real_adam_and_private_rng(tmp_path: Path) -> None:
    # Given: one real update and a complete atomic transaction.
    protocol, learner, examples = fixture(tmp_path)
    history = (Validation(update=0, correct=0, questions=2, nll=8.0),)
    store = Checkpoints(
        tmp_path / "run", protocol, tuple(e.example_id for e in examples)
    )
    _ = learner.update(examples, learning_rate=protocol.schedule.rate(1))
    saved = store.save(learner, history)
    resumed = BabiLearner(learner.graph, learner.config)
    # When: restore then execute the same next scheduled update.
    restored = store.load(resumed)
    left = learner.update(examples, learning_rate=protocol.schedule.rate(2))
    right = resumed.update(examples, learning_rate=protocol.schedule.rate(2))
    # Then: model, RNG, trace and exact exposures continue identically.
    assert restored == saved
    assert right == left
    assert learner.exposure_counts == resumed.exposure_counts
    assert torch.equal(learner.window_rng.get_state(), resumed.window_rng.get_state())
    for a, b in zip(
        learner.model.parameters(), resumed.model.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_corrupt_progress_when_load_does_not_mutate(tmp_path: Path) -> None:
    # Given: an immutable checkpoint followed by corruption of its manifest.
    protocol, learner, examples = fixture(tmp_path)
    store = Checkpoints(
        tmp_path / "run", protocol, tuple(e.example_id for e in examples)
    )
    history = (Validation(update=0, correct=0, questions=2, nll=8.0),)
    saved = store.save(learner, history)
    path = store.path(0) / "manifest.json"
    _ = path.write_text(
        saved.model_copy(
            update={"progress": saved.progress.model_copy(update={"next_lr": 0.7})}
        ).model_dump_json()
    )
    before = tuple(p.detach().clone() for p in learner.model.parameters())
    # When/Then: rejection precedes any tensor mutation.
    with pytest.raises(BabiError):
        _ = store.load(learner)
    assert learner.updates == 0
    for a, b in zip(before, learner.model.parameters(), strict=True):
        assert torch.equal(a, b)
