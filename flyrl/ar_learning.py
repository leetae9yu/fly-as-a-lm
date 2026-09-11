"""Typed model construction over the shared likelihood-learning engine."""

from typing import TypeAlias, assert_never

from flyrl.ar_config import ARConfig
from flyrl.ar_engine import TokenLearner
from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph
from flyrl.gru_model import GRULM
from flyrl.transformer_model import TransformerLM


class ARLearner(TokenLearner[ConnectomeLM]):
    """Preserve the original two-argument anatomical learner API."""

    def __init__(self, graph: Graph, config: ARConfig) -> None:
        """Initialize the historical anatomical model using shared optimization."""
        super().__init__(graph, config, ConnectomeLM(graph, config))


ExperimentLearner: TypeAlias = (
    ARLearner | TokenLearner[GRULM] | TokenLearner[TransformerLM]
)


def make_learner(graph: Graph, config: ARConfig) -> ExperimentLearner:
    """Construct exactly the declared architecture, without unused model allocation."""
    match config.architecture:
        case "connectome":
            return ARLearner(graph, config)
        case "gru":
            return TokenLearner(graph, config, GRULM(config))
        case "transformer":
            return TokenLearner(graph, config, TransformerLM(config))
        case _:
            assert_never(config.architecture)
