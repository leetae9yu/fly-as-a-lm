"""Quality panels use the real circuit and independent decoding expectations."""

import random
from copy import deepcopy
from dataclasses import FrozenInstanceError
from math import exp

import numpy as np
import pytest
import torch
from numpy.random.mtrand import get_bit_generator
from pydantic import ValidationError

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.bpe_data import BPECorpus, make_bpe_corpus
from flyrl.connectome import Graph
from flyrl.language_data import CorpusError, TextSplits
from flyrl.quality_generation import (
    QualityGenerationOptions,
    generate_quality_panel,
    has_repeated_word_block,
    nucleus_probabilities,
)


@pytest.fixture
def corpus() -> BPECorpus:
    return make_bpe_corpus(
        TextSplits(
            train="A cat and a dog played. " * 4,
            valid="A cat played.",
            test="A dog played.",
            provenance="synthetic generation test",
        ),
        vocab_size=257,
    )


@pytest.fixture
def learner(corpus: BPECorpus) -> ARLearner:
    source, target = np.nonzero(np.ones((6, 6)) - np.eye(6))
    graph = Graph(
        tuple(f"synthetic:{i}" for i in range(6)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic generation test, not anatomical data",
    )
    return ARLearner(
        graph,
        ARConfig(
            alphabet_size=len(corpus.vocabulary),
            tokenization="bpe",
            context=1,
            generation_context="windowed",
        ),
    )


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.01, [0.0, 1.0, 0.0, 0.0]),
        (0.5, [0.0, 1.0, 0.0, 0.0]),
        (0.51, [0.0, 2 / 3, 1 / 3, 0.0]),
        (0.75, [0.0, 2 / 3, 1 / 3, 0.0]),
        (0.8, [1 / 7, 4 / 7, 2 / 7, 0.0]),
        (1.0, [0.125, 0.5, 0.25, 0.125]),
    ],
)
def test_nucleus_is_smallest_prefix(p: float, expected: list[float]) -> None:
    # Given: exact binary probabilities, including a tie broken by token ID.
    logits = torch.tensor([1 / 8, 1 / 2, 1 / 4, 1 / 8], dtype=torch.float64).log()
    original = logits.clone()
    # When: restricting the descending distribution at and across boundaries.
    actual = nucleus_probabilities(
        logits, QualityGenerationOptions(temperature=1, nucleus_p=p)
    )
    # Then: the crossing token is included, but no token beyond it is retained.
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
    assert torch.equal(logits, original)


def test_temperature_precedes_nucleus() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    weights = [exp(value / 2) for value in (0, 1, 2)]
    total = sum(weights)
    expected = torch.tensor([value / total for value in weights], dtype=torch.float64)
    actual = nucleus_probabilities(
        logits, QualityGenerationOptions(temperature=2, nucleus_p=1)
    )
    torch.testing.assert_close(actual, expected)
    truncated = nucleus_probabilities(
        logits, QualityGenerationOptions(temperature=2, nucleus_p=0.6)
    )
    torch.testing.assert_close(
        truncated,
        torch.tensor(
            [0, weights[1] / sum(weights[1:]), weights[2] / sum(weights[1:])],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        _ = QualityGenerationOptions(temperature=temperature)


@pytest.mark.parametrize("p", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_nucleus(p: float) -> None:
    with pytest.raises(ValidationError):
        _ = QualityGenerationOptions(nucleus_p=p)


def _dense_logits(learner: ARLearner, history: list[int]) -> torch.Tensor:
    """Replay the anatomical equation independently of model.step and forward."""
    model = learner.model
    matrix = torch.zeros((model.nodes, model.nodes))
    matrix[model.edges[0], model.edges[1]] = model.weight.detach()
    state = torch.zeros((model.nodes, 1))
    for token in history:
        drive = torch.zeros_like(state)
        drive[model.sensory] = model.codes[token, :, None]
        state = (1 - model.config.leak) * state + model.config.leak * torch.tanh(
            matrix @ (state + drive) + model.bias[:, None] + drive
        )
    return (state[model.ports].T @ model.readout + model.output_bias)[0].detach()


def _expected_probabilities(
    logits: torch.Tensor, temperature: float, p: float
) -> torch.Tensor:
    weights = [exp(float(value.item()) / temperature) for value in logits]
    total = sum(weights)
    order = sorted(range(len(weights)), key=lambda index: -weights[index])
    retained: list[int] = []
    cumulative = 0.0
    for index in order:
        retained.append(index)
        cumulative += weights[index] / total
        if cumulative >= p:
            break
    denominator = sum(weights[index] for index in retained)
    return torch.tensor(
        [
            weight / denominator if index in retained else 0
            for index, weight in enumerate(weights)
        ]
    )


def test_panel_matches_independent_stateful_feedback(
    learner: ARLearner, corpus: BPECorpus
) -> None:
    # Given: prompt history exceeds the configured training/windowed context.
    prompts = ("A cat", "A dog")
    seeds = ((10_000, 10_001, 10_002), (10_100, 10_101, 10_102))
    options = QualityGenerationOptions(length=8, temperature=0.8, nucleus_p=0.9)
    records = generate_quality_panel(learner, corpus, prompts, seeds, options)
    assert len(records) == 8
    differs_from_reset = False
    sampled_paths: set[tuple[int, ...]] = set()
    for prompt_index, prompt in enumerate(prompts):
        for slot in range(4):
            record = records[4 * prompt_index + slot]
            seed = None if slot == 0 else seeds[prompt_index][slot - 1]
            rng = torch.Generator().manual_seed(0 if seed is None else seed)
            history = corpus.encode(prompt)
            generated: list[int] = []
            for _ in range(options.length):
                logits = _dense_logits(learner, history)
                differs_from_reset |= not torch.allclose(
                    logits, _dense_logits(learner, history[-1:])
                )
                probabilities = _expected_probabilities(
                    logits, options.temperature, options.nucleus_p
                )
                chosen = (
                    int(logits.argmax().item())
                    if seed is None
                    else int(torch.multinomial(probabilities, 1, generator=rng).item())
                )
                generated.append(chosen)
                history.append(chosen)
            assert record.prompt == prompt
            assert record.prompt_ids == tuple(corpus.encode(prompt))
            assert record.mode == ("greedy" if seed is None else "sampled")
            assert record.draw == (0 if seed is None else slot - 1)
            assert record.seed == seed
            assert record.token_ids == tuple(generated)
            assert record.text == corpus.decode(history)
            if seed is not None:
                sampled_paths.add(record.token_ids)
    assert differs_from_reset
    assert len(sampled_paths) > 1
    assert records == generate_quality_panel(learner, corpus, prompts, seeds, options)


def test_p_one_sampling_and_greedy_ignore_sampling_options(
    learner: ARLearner, corpus: BPECorpus
) -> None:
    prompt = "A cat"
    options = QualityGenerationOptions(length=10, temperature=2, nucleus_p=1)
    greedy, sampled = generate_quality_panel(
        learner, corpus, (prompt,), ((37,),), options
    )
    history = corpus.encode(prompt)
    rng = torch.Generator().manual_seed(37)
    for token in sampled.token_ids:
        probabilities = _expected_probabilities(_dense_logits(learner, history), 2, 1)
        assert token == int(torch.multinomial(probabilities, 1, generator=rng).item())
        history.append(token)
    other = generate_quality_panel(
        learner,
        corpus,
        (prompt,),
        ((),),
        QualityGenerationOptions(length=10, temperature=0.01, nucleus_p=0.01),
    )
    assert other == (greedy,)


def test_state_retention_changes_actual_greedy_decisions(
    learner: ARLearner, corpus: BPECorpus
) -> None:
    # Given: a real readout whose decision boundary separates full and reset state.
    prompt = "A cat"
    history = corpus.encode(prompt)
    with torch.no_grad():
        _ = learner.model.readout.zero_()
        _ = learner.model.output_bias.fill_(-100)
        learner.model.readout[:, 0] = 10
        learner.model.output_bias[0] = 0
        full = float(_dense_logits(learner, history)[0].item())
        reset = float(_dense_logits(learner, history[-1:])[0].item())
        assert abs(full - reset) > 0.01
        learner.model.output_bias[1] = (full + reset) / 2
    # When: generating with state retained despite the learner's windowed setting.
    (record,) = generate_quality_panel(
        learner, corpus, (prompt,), ((),), QualityGenerationOptions(length=8)
    )
    reset_choice = int(_dense_logits(learner, history[-1:]).argmax().item())
    assert record.token_ids[0] != reset_choice
    # Then: every later decision uses the actual generated token's full history.
    for chosen in record.token_ids:
        assert chosen == int(_dense_logits(learner, history).argmax().item())
        history.append(chosen)


def test_p_one_retains_tail_after_cumulative_rounding() -> None:
    logits = torch.tensor([0.0, -40.0], dtype=torch.float64)
    probabilities = nucleus_probabilities(
        logits, QualityGenerationOptions(temperature=1, nucleus_p=1)
    )
    assert probabilities[1].item() == pytest.approx(exp(-40), rel=1e-12, abs=0)


def test_generation_preserves_rng_parameters_and_progress(
    learner: ARLearner, corpus: BPECorpus
) -> None:
    python_rng = random.getstate()
    numpy_rng = deepcopy(get_bit_generator().state)
    torch_rng = torch.random.get_rng_state().clone()
    training_rng = learner.window_rng.get_state().clone()
    parameters = [
        parameter.detach().clone() for parameter in learner.model.parameters()
    ]
    records = generate_quality_panel(
        learner,
        corpus,
        ("A cat", "A dog"),
        ((11, 12), (13, 14)),
        QualityGenerationOptions(length=4),
    )
    assert random.getstate() == python_rng
    np.testing.assert_equal(get_bit_generator().state, numpy_rng)
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert torch.equal(learner.window_rng.get_state(), training_rng)
    assert learner.updates == 0
    assert learner.trace == []
    assert learner.model.training
    for old, parameter in zip(parameters, learner.model.parameters(), strict=True):
        assert torch.equal(old, parameter)
        assert parameter.grad is None
    assert isinstance(records, tuple)
    for name in ("text", "prompt", "seed", "token_ids"):
        with pytest.raises(FrozenInstanceError):
            setattr(records[0], name, "changed")


def test_zero_length_and_independent_prompts(
    learner: ARLearner, corpus: BPECorpus
) -> None:
    options = QualityGenerationOptions(length=0)
    records = generate_quality_panel(learner, corpus, ("A cat",), ((7, 8),), options)
    assert len(records) == 3
    assert all(record.token_ids == () and record.text == "A cat" for record in records)
    assert generate_quality_panel(learner, corpus, (), (), options) == ()
    options = QualityGenerationOptions(length=4)
    panel = generate_quality_panel(
        learner, corpus, ("A cat", "A dog"), ((7,), (8,)), options
    )
    alone = generate_quality_panel(learner, corpus, ("A dog",), ((8,),), options)
    assert panel[2:] == alone


def test_invalid_generation_boundaries(learner: ARLearner, corpus: BPECorpus) -> None:
    options = QualityGenerationOptions(length=1)
    with pytest.raises(ValidationError):
        _ = QualityGenerationOptions(length=-1)
    with pytest.raises(CorpusError, match="nonempty prompt"):
        _ = generate_quality_panel(learner, corpus, ("",), ((),), options)
    with pytest.raises(ValueError, match="seed"):
        _ = generate_quality_panel(learner, corpus, ("A cat",), (), options)
    with pytest.raises(ValueError, match="seed"):
        _ = generate_quality_panel(learner, corpus, ("A cat",), ((2**64,),), options)
    different = make_bpe_corpus(
        TextSplits(train="cat " * 20, valid="cat", test="cat", provenance="test"),
        vocab_size=260,
    )
    with pytest.raises(CorpusError, match="vocabulary"):
        _ = generate_quality_panel(learner, different, ("cat",), ((),), options)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("123 ...", False),
        ("go go", False),
        ("go GO Go!", True),
        ("we can't, we can't; WE CAN'T.", True),
        ("a b a b a b", True),
        ("a b a b c a b", False),
        ("cat concatenate cat", False),
        ("one two three four five six seven eight nine ten " * 3, True),
        ("one two three four five six seven eight nine ten eleven " * 3, False),
        ("start a b a b a b end", True),
        ("a b a b x a b", False),
    ],
)
def test_repeated_word_blocks(text: str, *, expected: bool) -> None:
    assert has_repeated_word_block(text) is expected
