"""Validated boundaries and machine-readable metrics for sparse likelihood learning."""

from typing import Annotated, Final, Literal, Self, TypeAlias, assert_never

from pydantic import Field, model_validator

from flyrl.language_data import MAX_SYMBOLS
from flyrl.language_models import Settings

Tokenization: TypeAlias = Literal["character", "bpe"]
MIN_BYTE_VOCABULARY: Final = 257


class ARConfig(Settings):
    """Update-independent configuration; checkpoints require exact equality."""

    alphabet_size: Annotated[int, Field(ge=2, le=65536)]
    tokenization: Tokenization = "character"
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    device: str = "cpu"
    control: Literal["real", "shuffled", "frozen"] = "real"
    context: Annotated[int, Field(ge=1)] = 32
    batch_size: Annotated[int, Field(ge=1)] = 8
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.003
    weight_decay: Annotated[float, Field(ge=0)] = 0.0
    gradient_clip: Annotated[float, Field(gt=0)] = 1.0
    leak: Annotated[float, Field(gt=0, le=1)] = 0.5
    initial_gain: Annotated[float, Field(gt=0, le=1)] = 0.9
    readout_neurons: Annotated[int, Field(ge=1, le=1024)] = 256
    edge_chunk: Annotated[int, Field(ge=1)] = 65536
    eval_windows: Annotated[int, Field(ge=1)] = 512
    sample_length: Annotated[int, Field(ge=0)] = 120

    @model_validator(mode="after")
    def vocabulary_matches_unit(self) -> Self:
        """Keep legacy character bounds and require a complete byte alphabet for BPE."""
        match self.tokenization:
            case "character":
                if self.alphabet_size > MAX_SYMBOLS:
                    message = "Character vocabularies are limited to 48 symbols"
                    raise ValueError(message)
            case "bpe":
                if self.alphabet_size < MIN_BYTE_VOCABULARY:
                    message = "Byte-level BPE requires 256 bytes and an unknown token"
                    raise ValueError(message)
            case _:
                assert_never(self.tokenization)
        return self


class ARMetrics(Settings):
    """Final-target metrics on exactly the baseline's disjoint windows."""

    windows: int
    greedy_accuracy: float
    nll: float
    bits_per_character: float | None = None
    bits_per_token: float | None = None
    perplexity: float | None = None


class TraceEntry(Settings):
    """All-position training loss before an optimizer update."""

    update: int
    nll: float
    gradient_norm: float
