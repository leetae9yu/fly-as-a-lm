"""Validated boundaries and machine-readable metrics for sparse likelihood learning."""

from typing import Annotated, Final, Literal, Self, TypeAlias, assert_never

from pydantic import Field, model_validator

from flyrl.language_data import MAX_SYMBOLS
from flyrl.language_models import Settings

Tokenization: TypeAlias = Literal["character", "bpe"]
Architecture: TypeAlias = Literal["connectome", "gru", "transformer"]
GenerationContext: TypeAlias = Literal["stateful", "windowed"]
PortPolicy: TypeAlias = Literal[
    "legacy_random",
    "alpn_mbon",
    "alpn_random",
    "random_mbon",
    "random_random",
]
MIN_BYTE_VOCABULARY: Final = 257
PORT_IDENTITY_FIELDS: Final = {
    "port_policy",
    "port_manifest_sha256",
    "sensory_indices",
    "readout_indices",
}


class ARConfig(Settings):
    """Update-independent configuration; checkpoints require exact equality."""

    alphabet_size: Annotated[int, Field(ge=2, le=65536)]
    tokenization: Tokenization = "character"
    architecture: Architecture = "connectome"
    trainable_codes: bool = False
    generation_context: GenerationContext = "stateful"
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
    port_policy: PortPolicy = "legacy_random"
    port_manifest_sha256: Annotated[str, Field(min_length=64, max_length=64)] | None = (
        None
    )
    sensory_indices: tuple[int, ...] | None = None
    readout_indices: tuple[int, ...] | None = None

    @property
    def condition(self) -> str:
        """Use distinct output names for anatomical controls and dense models."""
        match self.architecture:
            case "connectome":
                return self.control
            case "gru" | "transformer":
                return self.architecture
            case _:
                assert_never(self.architecture)

    @property
    def is_anatomical(self) -> bool:
        """Distinguish graph controls from ordinary dense model references."""
        match self.architecture:
            case "connectome":
                return True
            case "gru" | "transformer":
                return False
            case _:
                assert_never(self.architecture)

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

    @model_validator(mode="after")
    def architecture_matches_control(self) -> Self:
        """Dense references have no anatomical ablation or unbounded state API."""
        match self.architecture:
            case "connectome":
                pass
            case "gru" | "transformer":
                if self.trainable_codes:
                    message = "Trainable sensory codes require connectome architecture"
                    raise ValueError(message)
                if self.control != "real" or self.generation_context != "windowed":
                    message = (
                        "Dense baselines require real control and windowed generation"
                    )
                    raise ValueError(message)
                if self.port_policy != "legacy_random":
                    message = "Dense baselines cannot use anatomical port policies"
                    raise ValueError(message)
            case _:
                assert_never(self.architecture)
        return self

    @model_validator(mode="after")
    def explicit_ports_match_policy(self) -> Self:
        """Require complete, disjoint explicit indices for nonlegacy policies."""
        match self.port_policy:
            case "legacy_random":
                if (
                    self.port_manifest_sha256 is not None
                    or self.sensory_indices is not None
                    or self.readout_indices is not None
                ):
                    message = "Legacy random ports cannot carry explicit port metadata"
                    raise ValueError(message)
            case "alpn_mbon" | "alpn_random" | "random_mbon" | "random_random":
                if (
                    self.port_manifest_sha256 is None
                    or self.sensory_indices is None
                    or self.readout_indices is None
                ):
                    message = "Explicit port policies require manifest and index tuples"
                    raise ValueError(message)
                if (
                    len(self.readout_indices) != self.readout_neurons
                    or not self.sensory_indices
                    or len(set(self.sensory_indices)) != len(self.sensory_indices)
                    or len(set(self.readout_indices)) != len(self.readout_indices)
                    or bool(set(self.sensory_indices) & set(self.readout_indices))
                    or min(self.sensory_indices) < 0
                    or min(self.readout_indices) < 0
                ):
                    message = (
                        "Explicit sensory/readout indices must be sized and disjoint"
                    )
                    raise ValueError(message)
            case _:
                assert_never(self.port_policy)
        return self


def parameter_identity_json(config: ARConfig) -> str:
    """Serialize parameters compatibly while binding every explicit port."""
    match config.port_policy:
        case "legacy_random":
            return config.model_dump_json(exclude=PORT_IDENTITY_FIELDS)
        case "alpn_mbon" | "alpn_random" | "random_mbon" | "random_random":
            return config.model_dump_json()
        case _:
            assert_never(config.port_policy)


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
