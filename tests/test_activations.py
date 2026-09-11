"""Activation recordings must match the actual generation decisions."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter, ValidationError

from flyrl.activations import ActivationMetadata, ActivationOptions, export_activations
from flyrl.ar_config import ARConfig, GenerationContext
from flyrl.ar_learning import ARLearner
from flyrl.bpe_data import BPECorpus, make_bpe_corpus
from flyrl.connectome import Graph
from flyrl.language_data import TextSplits


@pytest.fixture
def corpus() -> BPECorpus:
    return make_bpe_corpus(
        TextSplits(
            train="A cat saw a red ball. A cat saw a blue ball. " * 4,
            valid="A dog saw a ball.",
            test="A bird saw a cat.",
            provenance="original synthetic test stories, not TinyStories",
        ),
        vocab_size=280,
    )


@pytest.fixture
def graph() -> Graph:
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    return Graph(
        tuple(f"test-neuron:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic test graph, not anatomical data",
    )


@pytest.mark.parametrize("mode", ["stateful", "windowed"])
@pytest.mark.parametrize("greedy", [True, False])
def test_recording_matches_generation_without_mutation(
    graph: Graph,
    corpus: BPECorpus,
    tmp_path: Path,
    mode: GenerationContext,
    *,
    greedy: bool,
) -> None:
    # Given: the same initialized circuit and isolated generation RNG.
    learner = ARLearner(
        graph,
        ARConfig(
            alphabet_size=len(corpus.vocabulary),
            tokenization="bpe",
            generation_context=mode,
            context=3,
        ),
    )
    prompt = corpus.encode("A cat")
    expected = learner.generate(prompt, 4, greedy=greedy)
    parameters = [p.detach().clone() for p in learner.model.parameters()]
    rng = learner.window_rng.get_state().clone()
    # When: exporting the states that selected those tokens.
    path = export_activations(
        learner,
        corpus,
        "A cat",
        tmp_path,
        ActivationOptions(length=4, greedy=greedy, max_neurons=5),
    )
    # Then: the export preserves token identity and does not train or consume RNG.
    metadata = ActivationMetadata.model_validate_json(path.read_text())
    with (
        (tmp_path / "activations.npz").open("rb") as stream,
        NpzFile[np.generic](stream, allow_pickle=False) as data,
    ):
        assert data["generated_ids"].tolist() == expected
        assert data["states"].shape == (4, 12)
        assert data["node_ids"].tolist() == list(graph.node_ids)
        strengths = TypeAdapter(list[float]).validate_python(
            np.mean(
                np.abs(np.asarray(data["states"], dtype=np.float32)), axis=0
            ).tolist()
        )
        expected_indices = sorted(range(12), key=lambda i: -strengths[i])[:5]
        assert data["selected_indices"].tolist() == expected_indices
    assert metadata.corpus_fingerprint == corpus.fingerprint
    assert metadata.generated_text == corpus.decode(prompt + expected)
    assert torch.equal(rng, learner.window_rng.get_state())
    for previous, current in zip(parameters, learner.model.parameters(), strict=True):
        assert torch.equal(previous, current)
    assert (tmp_path / "activations-001.png").is_file()
    assert (tmp_path / "activations-001.svg").is_file()


def test_state_is_before_chosen_token_is_fed_back(
    graph: Graph, corpus: BPECorpus, tmp_path: Path
) -> None:
    # Given: a known prompt and its hand-replayed recurrent state.
    learner = ARLearner(
        graph,
        ARConfig(alphabet_size=len(corpus.vocabulary), tokenization="bpe"),
    )
    state = torch.zeros((12, 1))
    with torch.no_grad():
        for token in corpus.encode("A cat"):
            _, state = learner.model.step(torch.tensor([token]), state)
    # When: predicting one token.
    _ = export_activations(
        learner, corpus, "A cat", tmp_path, ActivationOptions(length=1)
    )
    # Then: recording excludes feedback from the token it purports to explain.
    with (
        (tmp_path / "activations.npz").open("rb") as stream,
        NpzFile[np.generic](stream, allow_pickle=False) as data,
    ):
        recorded = torch.tensor(np.asarray(data["states"], dtype=np.float32))
        assert torch.equal(recorded[0], state[:, 0])
        with torch.no_grad():
            logits = state[learner.model.ports].T @ learner.model.readout
            logits += learner.model.output_bias
            token = TypeAdapter(int).validate_python(data["generated_ids"][0])
            expected = logits.softmax(dim=1)[0, token].item()
        assert data["probabilities"][0] == pytest.approx(expected)


@pytest.mark.parametrize(("length", "max_neurons"), [(0, 5), (1, 0)])
def test_invalid_export_dimensions_leave_no_artifacts(
    graph: Graph,
    corpus: BPECorpus,
    tmp_path: Path,
    length: int,
    max_neurons: int,
) -> None:
    # Given: invalid caller-controlled export dimensions.
    learner = ARLearner(
        graph, ARConfig(alphabet_size=len(corpus.vocabulary), tokenization="bpe")
    )
    # When/Then: rejecting them occurs before writing partial output.
    with pytest.raises(ValidationError):
        _ = export_activations(
            learner,
            corpus,
            "A",
            tmp_path,
            ActivationOptions(length=length, max_neurons=max_neurons),
        )
    assert list(tmp_path.iterdir()) == []
