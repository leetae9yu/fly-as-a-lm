"""Portable exact neuron-group manifests for regional probes."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from flyrl.connectome import Graph
from scripts.anatomy_port_artifacts import AnatomyIndices
from scripts.regional_probe_groups import (
    SELECTION_SEED,
    ProbeGroupName,
    RegionalSelection,
    regional_probe_groups,
)


class ProbeGroupRecord(BaseModel):
    """One exact capacity-matched probe input."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: ProbeGroupName
    draw: int
    indices: tuple[int, ...]
    node_ids: tuple[str, ...]


class SeedGroupPlan(BaseModel):
    """All predeclared groups for one paired real/shuffled checkpoint seed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    sensory_indices: tuple[int, ...]
    trained_readout_indices: tuple[int, ...]
    groups: tuple[ProbeGroupRecord, ...]


class RegionalGroupManifest(BaseModel):
    """Hash-bound exact group membership frozen before probe fitting."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-regional-probe-groups-v1"] = "flyrl-regional-probe-groups-v1"
    graph_sha256: str
    port_manifest_sha256: str
    port_arrays_sha256: str
    selection_seed: int
    group_size: int
    draws: int
    seeds: tuple[SeedGroupPlan, ...]


@dataclass(frozen=True, slots=True)
class GroupManifestIdentities:
    """Immutable graph and anatomy artifact hashes."""

    graph_sha256: str
    port_manifest_sha256: str
    port_arrays_sha256: str
    group_size: int = 97
    draws: int = 5


def build_group_manifest(
    graph: Graph,
    anatomy: AnatomyIndices,
    selections: tuple[RegionalSelection, ...],
    identities: GroupManifestIdentities,
) -> RegionalGroupManifest:
    """Build all exact paired-seed groups before any probe metric exists."""
    if (
        identities.group_size < 1
        or identities.draws < 1
        or not selections
        or len({selection.seed for selection in selections}) != len(selections)
        or any(
            (
                selection.alpn,
                selection.mbon,
                selection.kenyon,
            )
            != (anatomy.alpn, anatomy.mbon, anatomy.kenyon)
            for selection in selections
        )
    ):
        message = "Regional group source selections are incomplete or inconsistent"
        raise ValueError(message)
    plans: list[SeedGroupPlan] = []
    for selection in sorted(selections, key=lambda item: item.seed):
        groups = regional_probe_groups(
            graph,
            selection,
            group_size=identities.group_size,
            draws=identities.draws,
        )
        plans.append(
            SeedGroupPlan(
                seed=selection.seed,
                sensory_indices=selection.sensory,
                trained_readout_indices=selection.trained_readout,
                groups=tuple(
                    ProbeGroupRecord(
                        name=group.name,
                        draw=group.draw,
                        indices=group.indices,
                        node_ids=tuple(
                            graph.node_ids[index] for index in group.indices
                        ),
                    )
                    for group in groups
                ),
            )
        )
    return RegionalGroupManifest(
        graph_sha256=identities.graph_sha256,
        port_manifest_sha256=identities.port_manifest_sha256,
        port_arrays_sha256=identities.port_arrays_sha256,
        selection_seed=SELECTION_SEED,
        group_size=identities.group_size,
        draws=identities.draws,
        seeds=tuple(plans),
    )
