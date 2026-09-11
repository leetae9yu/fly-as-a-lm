"""Acquisition stops at complete delimiters within a hard byte budget."""

import importlib.util
from io import BytesIO

import pytest

from flyrl.language_data import CorpusError
from flyrl.story_fetch import PrefixBudget, read_prefix


def test_bounded_fetch_contract_exists() -> None:
    # Given/When: resolving acquisition independently of model dependencies.
    spec = importlib.util.find_spec("flyrl.story_fetch")
    # Then: explicit bounded acquisition is available.
    assert spec is not None


def test_prefix_reader_stops_after_complete_stories() -> None:
    # Given: a stream with a multi-line story and additional unwanted data.
    expected = b"One\nstory.\n<|endoftext|>\n"
    stream = BytesIO(expected + b"Another.\n<|endoftext|>\n")
    # When: selecting just one complete story.
    result = read_prefix(
        iter(lambda: stream.read(1), b""), PrefixBudget(stories=1, max_bytes=100)
    )
    # Then: no later bytes were read or retained.
    assert result.payload == expected
    assert result.bytes_read == len(expected)
    assert stream.tell() == len(expected)


def test_prefix_reader_rejects_partial_or_over_budget_story() -> None:
    # Given/When/Then: incomplete EOF never becomes a fetched story.
    with pytest.raises(CorpusError, match="complete"):
        _ = read_prefix(iter((b"unfinished",)), PrefixBudget(stories=1, max_bytes=100))
    stream = BytesIO(b"too large\n<|endoftext|>\n")
    with pytest.raises(CorpusError, match="byte budget"):
        _ = read_prefix(
            iter(lambda: stream.read(1), b""), PrefixBudget(stories=1, max_bytes=5)
        )
    assert stream.tell() <= 5
