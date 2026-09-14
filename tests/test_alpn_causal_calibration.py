"""Exact directed calibration and old-training-only state statistics."""

from itertools import accumulate
from pathlib import Path
from random import getstate
from typing import Literal, Never

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_model import ConnectomeLM, IndexArray
from flyrl.connectome import Graph
from flyrl.story_data import StorySplit
from scripts.alpn_causal_calibration import (
    OLD_TRAINING_TARGETS,
    CalibrationConfig,
    TrainingActivity,
    directed_distances,
    original_graph_coordinates,
    outgoing_weight_l2,
    paired_coordinates,
    training_activity,
)
from scripts.alpn_causal_groups import FloatMatrix


def graph_fixture() -> Graph:
    return Graph(
        tuple(str(i) for i in range(7)),
        np.array([2, 0, 4, 1, 0, 3, 4], dtype=np.int64),
        np.array([3, 1, 4, 2, 2, 4, 0], dtype=np.int64),
        np.array([-3.0, 2.0, -5.0, 4.0, 1.0, 6.0, 7.0]),
        "directed cycle, shortcut, self loop and isolated nodes",
    )


def model_fixture(wiring: Literal["real", "shuffled"]) -> ConnectomeLM:
    model = ConnectomeLM(
        graph_fixture(),
        ARConfig(
            alphabet_size=3,
            control=wiring,
            seed=6,
            leak=1.0,
            port_policy="random_random",
            port_manifest_sha256="a" * 64,
            sensory_indices=(0,),
            readout_indices=(4,),
            readout_neurons=1,
        ),
    )
    with torch.no_grad():
        _ = model.weight.copy_(torch.tensor([0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75]))
        _ = model.bias.copy_(torch.tensor([0.0, -0.2, 0.3, -0.4, 0.5, 10.0, -10.0]))
        _ = model.codes.copy_(torch.tensor([[4.0], [0.0], [-4.0]]))
    return model


def dense_states(model: ConnectomeLM) -> FloatMatrix:
    tokens = (0, 1, 2, 1, 0, 2, 2, 0, 1, 2)
    matrix = torch.zeros((model.nodes, model.nodes))
    with torch.no_grad():
        for edge in range(model.weight.numel()):
            target, source = (int(model.edges[i, edge].item()) for i in range(2))
            matrix[target, source] += model.weight[edge]
        states: list[torch.Tensor] = []
        for start, end in ((0, 5), (5, 7), (7, 10)):
            state = torch.zeros(model.nodes)
            for token in tokens[start : end - 1]:
                drive = torch.zeros(model.nodes)
                drive[model.sensory] = model.codes[token]
                state = torch.tanh(matrix @ (state + drive) + model.bias + drive)
                states.append(state)
    return np.asarray(torch.stack(states).numpy(), dtype=np.float64)


def test_original_structure_uses_signed_graph_not_learned_weights() -> None:
    graph = graph_fixture()
    before = tuple(array.copy() for array in (graph.source, graph.target, graph.weight))
    dense: FloatMatrix = np.zeros((7, 7), dtype=np.float64)
    dense[graph.target, graph.source] = graph.weight
    expected = np.empty((7, 4), dtype=np.float64)
    for column, axis in enumerate((1, 0)):
        expected[:, column] = (np.abs(dense) > 0).sum(axis=axis)
        expected[:, column + 2] = np.abs(dense).sum(axis=axis)
    actual = original_graph_coordinates(graph)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    for array, snapshot in zip(
        (graph.source, graph.target, graph.weight), before, strict=True
    ):
        np.testing.assert_array_equal(array, snapshot)


@pytest.mark.parametrize("wiring", ["real", "shuffled"])
def test_parameter_order_l2_and_actual_endpoint_distances(
    wiring: Literal["real", "shuffled"],
) -> None:
    model = model_fixture(wiring)
    edges: IndexArray = np.asarray(model.edges.numpy(), dtype=np.int64)
    weights = np.asarray(model.weight.detach().numpy(), dtype=np.float64)
    expected = np.zeros(7, dtype=np.float64)
    distances: FloatMatrix = np.full((7, 7), np.inf)
    np.fill_diagonal(distances, 0)
    for edge, weight in enumerate(weights.flat):
        target, source = int(edges.item(0, edge)), int(edges.item(1, edge))
        expected[source] += float(weight) ** 2
        distances[source, target] = min(float(distances.item(source, target)), 1)
    for middle in range(7):
        distances = np.minimum(
            distances, distances[:, middle, None] + distances[None, middle, :]
        )
    actual = outgoing_weight_l2(model)
    np.testing.assert_array_equal(actual, np.sqrt(expected))
    paths = directed_distances(model)
    np.testing.assert_array_equal(
        paths, np.minimum(np.asarray((distances[0], distances[:, 4])).T, 3)
    )
    assert paths.dtype == actual.dtype == np.float64
    if wiring == "real":
        np.testing.assert_array_equal(
            paths, [[0, 3], [1, 3], [1, 2], [2, 1], [3, 0], [3, 3], [3, 3]]
        )
    else:
        assert bool((np.abs(edges[0:1].reshape(-1) - graph_fixture().target) > 0).any())
        assert model.topology.has_duplicates


def test_multiple_ports_and_zero_weights_keep_anatomical_paths() -> None:
    model = model_fixture("real")
    model.sensory = torch.tensor([0, 5])
    model.ports = torch.tensor([4, 6])
    with torch.no_grad():
        _ = model.weight.zero_()
    np.testing.assert_array_equal(outgoing_weight_l2(model), np.zeros(7))
    np.testing.assert_array_equal(
        directed_distances(model),
        [[0, 3], [1, 3], [1, 2], [2, 1], [3, 0], [0, 3], [3, 0]],
    )


@pytest.mark.parametrize(("chunk", "batch"), [(1, 1), (2, 2), (64, 5)])
@pytest.mark.parametrize("wiring", ["real", "shuffled"])
def test_activity_exact_dense_and_source_immutability(
    chunk: int,
    batch: int,
    wiring: Literal["real", "shuffled"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_fixture(wiring)
    _ = model.train(wiring == "real")
    model.topology.training = not model.training
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    values = tuple(
        value.detach().clone() for value in (*model.parameters(), *model.buffers())
    )
    modes = tuple(module.training for module in model.modules())
    rng, python_rng = torch.get_rng_state().clone(), getstate()
    tokens = np.array([0, 1, 2, 1, 0, 2, 2, 0, 1, 2, 0], dtype=np.int64)
    split = StorySplit(
        offsets=(0, 5, 5, 7, 10, 11), sha256=("a", "empty", "b", "c", "singleton")
    )
    indices = (6, 3, 0, 5)
    dense = dense_states(model)[:, list(indices)]

    def forbidden(*_args: str, **_kwargs: str) -> Never:
        message = "Calibration must not open any split, artifact or RNG"
        raise AssertionError(message)

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    config = CalibrationConfig(
        chunk_size=chunk, story_batch_size=batch, expected_targets=7
    )
    result = training_activity(model, tokens, split, indices, config=config)
    expected = np.asarray(
        (dense.mean(axis=0), dense.std(axis=0), (np.abs(dense) >= 0.99).mean(axis=0))
    ).T
    np.testing.assert_array_equal(result.values, expected)
    assert result.indices == indices
    assert result.positions == 7
    assert result.values.dtype == np.float64
    assert not result.values.flags.writeable
    assert modes == tuple(module.training for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert python_rng == getstate()
    for value, snapshot in zip(
        (*model.parameters(), *model.buffers()), values, strict=True
    ):
        assert torch.equal(value, snapshot)
    assert all(
        parameter.grad is not None and bool((parameter.grad == 1).all())
        for parameter in model.parameters()
    )


def test_complete_old_training_count_and_default_rejects_subset() -> None:
    model = model_fixture("real")
    with torch.no_grad():
        _ = model.weight.zero_()
        _ = model.codes.zero_()
    lengths = [231] * 745 + [230] * 255
    offsets = tuple(accumulate(lengths, initial=0))
    split = StorySplit(offsets=offsets, sha256=tuple(str(i) for i in range(1000)))
    tokens = np.zeros(sum(lengths), dtype=np.int64)
    result = training_activity(
        model,
        tokens,
        split,
        (6, 0, 5),
        config=CalibrationConfig(chunk_size=64, story_batch_size=1000),
    )
    assert result.positions == OLD_TRAINING_TARGETS == 229745
    np.testing.assert_array_equal(result.values, [[-1, 0, 1], [0, 0, 0], [1, 0, 1]])
    with pytest.raises(ValueError, match="target"):
        _ = training_activity(
            model, tokens[:2], StorySplit(offsets=(0, 2), sha256=("a",)), (0,)
        )


def test_paired_assembly_order_and_selected_row_order() -> None:
    graph = graph_fixture()
    real, shuffled = model_fixture("real"), model_fixture("shuffled")
    indices = (6, 0, 3)
    first = TrainingActivity(
        indices, 7, np.array([[-1.0, 0.0, 1.0], [0.2, 0.3, 0.4], [-0.5, 0.6, 0.7]])
    )
    second = TrainingActivity(
        indices, 7, np.array([[1.0, 0.0, 1.0], [-0.2, 0.8, 0.9], [0.5, 0.1, 0.0]])
    )
    actual = paired_coordinates(graph, real, shuffled, (first, second))
    expected = np.empty((3, 16), dtype=np.float64)
    expected[:, :4] = np.log1p(original_graph_coordinates(graph)[list(indices)])
    expected[:, 4:] = np.asarray(
        (
            np.log1p(outgoing_weight_l2(real)[list(indices)]),
            np.log1p(outgoing_weight_l2(shuffled)[list(indices)]),
            directed_distances(real)[list(indices), 0],
            directed_distances(shuffled)[list(indices), 0],
            directed_distances(real)[list(indices), 1],
            directed_distances(shuffled)[list(indices), 1],
            first.values[:, 0],
            second.values[:, 0],
            np.log(np.maximum(first.values[:, 1], 0.001)),
            np.log(np.maximum(second.values[:, 1], 0.001)),
            first.values[:, 2],
            second.values[:, 2],
        )
    ).T
    assert actual.shape == (3, 16)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    for invalid in (
        TrainingActivity((0, 3, 6), 7, second.values),
        TrainingActivity(indices, 6, second.values),
    ):
        with pytest.raises(ValueError, match="activity"):
            _ = paired_coordinates(graph, real, shuffled, (first, invalid))


@pytest.mark.parametrize("indices", [(), (0, 0), (-1,), (7,)])
def test_invalid_activity_indices(indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="indices"):
        _ = training_activity(
            model_fixture("real"),
            np.array([0, 1], dtype=np.int64),
            StorySplit(offsets=(0, 2), sha256=("a",)),
            indices,
            config=CalibrationConfig(expected_targets=1),
        )
