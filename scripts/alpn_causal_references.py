"""Validate and package the exact frozen ALPN/readout heads without scoring.

The caller owns the destination and serializes the returned metric-free manifest.
ZIP names are flat; manifest paths retain the original source directory for the
published digest: ordered UTF-8 path bytes followed by raw SHA256 bytes.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, ClassVar, Final, Literal, Self
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.connectome_source import file_digest
from scripts.regional_probe_artifacts import load_head
from scripts.regional_probe_manifest import RegionalGroupManifest
from scripts.regional_probe_protocol import RegionalProbeProtocol

RESULT_SHA256: Final = (
    "ac7434b9f4ae313e48005569ab18a659a930c731e422fe96c09bc748a86a73ca"
)
PROTOCOL_SHA256: Final = (
    "6b134d0e928caacf13900ccf793a65954e02dadef6e40eee014fa7feb9bb0dba"
)
ORDERED_SHA256: Final = (
    "0d0ac287ebc406284ace0c5ce2467105541a48e7d94907b289c8d9b28c210f17"
)
TOTAL_BYTES: Final = 103_987_229
SOURCE_KEYS: Final = tuple((s, w) for s in range(7, 13) for w in ("real", "shuffled"))
STEMS: Final = (*(f"alpn-{draw}" for draw in range(5)), "trained_readout-0")
PATHS: Final = tuple(
    f"seed-{seed}-{wiring}/{stem}.{ext}"
    for seed, wiring in SOURCE_KEYS
    for stem in STEMS
    for ext in ("json", "npz")
)
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FrozenRecord(BaseModel):
    """Strict, immutable records containing no prior evaluation metrics."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ReferenceFile(FrozenRecord):
    """One unchanged JSON or NPZ payload and its original digest path."""

    path: str = Field(
        pattern=(
            r"^seed-(?:[7-9]|1[0-2])-(?:real|shuffled)/"
            r"(?:alpn-[0-4]|trained_readout-0)\.(?:json|npz)$"
        )
    )
    sha256: Digest
    size: int = Field(gt=0)

    @property
    def archive_name(self) -> str:
        """Return the unambiguous flat name used in the destination ZIP."""
        return self.path.replace("/", "-")


class ReferenceManifest(FrozenRecord):
    """Exactly sixty ALPN and twelve trained-readout heads, in published order."""

    format: Literal["flyrl-alpn-causal-references-v1"] = (
        "flyrl-alpn-causal-references-v1"
    )
    result_sha256: Digest
    protocol_sha256: Digest
    files: tuple[ReferenceFile, ...]

    @property
    def ordered_sha256(self) -> str:
        """Hash original paths followed by binary file digests, without separators."""
        digest = sha256()
        for item in self.files:
            digest.update(item.path.encode("utf-8"))
            digest.update(bytes.fromhex(item.sha256))
        return digest.hexdigest()

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Reject changed identities, omissions, duplicates and reordered files."""
        if (
            self.result_sha256 != RESULT_SHA256
            or self.protocol_sha256 != PROTOCOL_SHA256
            or tuple(item.path for item in self.files) != PATHS
            or sum(item.size for item in self.files) != TOTAL_BYTES
            or self.ordered_sha256 != ORDERED_SHA256
        ):
            message = "Frozen reference manifest identity or completeness differs"
            raise ValueError(message)
        return self


def _members(archive: ZipFile, expected: tuple[str, ...]) -> None:
    names = archive.namelist()
    if len(names) != len(expected) or set(names) != set(expected):
        message = "Reference archive membership differs (extra, missing or nested)"
        raise ValueError(message)


def _inputs(
    archive: ZipFile, groups_path: Path
) -> tuple[RegionalProbeProtocol, RegionalGroupManifest]:
    _members(
        archive,
        (
            "protocol.json",
            "summary.json",
            *(f"seed-{seed}-{wiring}-probes.zip" for seed, wiring in SOURCE_KEYS),
        ),
    )
    payload = archive.read("protocol.json")
    if sha256(payload).hexdigest() != PROTOCOL_SHA256:
        message = "Regional protocol identity differs"
        raise ValueError(message)
    protocol = RegionalProbeProtocol.model_validate_json(payload)
    group_bytes = groups_path.read_bytes()
    if sha256(group_bytes).hexdigest() != protocol.group_manifest_sha256:
        message = "Regional group identity differs"
        raise ValueError(message)
    groups = RegionalGroupManifest.model_validate_json(group_bytes)
    if (
        tuple((source.seed, source.wiring) for source in protocol.checkpoints)
        != SOURCE_KEYS
        or any(
            source.filename != f"seed-{source.seed}-{source.wiring}-random_random.zip"
            for source in protocol.checkpoints
        )
        or tuple(plan.seed for plan in groups.seeds) != tuple(range(7, 13))
    ):
        message = "Regional source identity differs"
        raise ValueError(message)
    return protocol, groups


def _collect(
    archive: ZipFile,
    protocol: RegionalProbeProtocol,
    groups: RegionalGroupManifest,
    directory: Path,
) -> tuple[ReferenceFile, ...]:
    records: list[ReferenceFile] = []
    for source in protocol.checkpoints:
        prefix = f"seed-{source.seed}-{source.wiring}"
        plan = next(plan for plan in groups.seeds if plan.seed == source.seed)
        output = directory / prefix
        output.mkdir()
        config = protocol.probe.model_copy(
            update={"seed": protocol.probe.seed + source.seed}
        )
        with ZipFile(BytesIO(archive.read(f"{prefix}-probes.zip"))) as inner:
            _members(
                inner,
                (
                    "source.json",
                    "heldout.npz",
                    *(
                        f"{group.name}-{group.draw}.{ext}"
                        for group in plan.groups
                        for ext in ("json", "npz")
                    ),
                ),
            )
            for stem in STEMS:
                group = next(
                    group
                    for group in plan.groups
                    if f"{group.name}-{group.draw}" == stem
                )
                for ext in ("json", "npz"):
                    name = f"{stem}.{ext}"
                    payload = inner.read(name)
                    _ = (output / name).write_bytes(payload)
                    records.append(
                        ReferenceFile(
                            path=f"{prefix}/{name}",
                            sha256=sha256(payload).hexdigest(),
                            size=len(payload),
                        )
                    )
                try:
                    head, _ = load_head(output / f"{stem}.json")
                except (ValueError, OSError, BadZipFile):
                    message = "Frozen reference head failed JSON/NPZ validation"
                    raise ValueError(message) from None
                if (
                    head.source != source
                    or head.group != group
                    or head.result.config != config
                    or head.extraction != protocol.extraction
                ):
                    message = (
                        "Frozen reference head source/group/config identity differs"
                    )
                    raise ValueError(message)
    return tuple(records)


def validate_reference_archive(path: Path, manifest: ReferenceManifest) -> None:
    """Check every byte against the pinned manifest, including CRC and member order."""
    manifest = ReferenceManifest.model_validate(manifest)
    with ZipFile(path) as archive:
        expected = tuple(item.archive_name for item in manifest.files)
        _members(archive, expected)
        if tuple(archive.namelist()) != expected:
            message = "Reference archive member order differs"
            raise ValueError(message)
        for item in manifest.files:
            payload = archive.read(item.archive_name)
            if len(payload) != item.size or sha256(payload).hexdigest() != item.sha256:
                message = "Reference archive payload identity differs"
                raise ValueError(message)


def package_references(
    regional_zip: Path, groups_path: Path, destination: Path
) -> ReferenceManifest:
    """Validate frozen inputs, then exclusively publish a deterministic flat ZIP.

    Only protocol and selected heads are parsed. Summary, source reports, other
    heads and heldout arrays remain unread. Existing destinations are never changed.
    """
    if file_digest(regional_zip) != RESULT_SHA256:
        message = "Regional result ZIP identity differs"
        raise ValueError(message)
    with (
        TemporaryDirectory(dir=destination.parent) as temporary,
        ZipFile(regional_zip) as archive,
    ):
        directory = Path(temporary)
        protocol, groups = _inputs(archive, groups_path)
        manifest = ReferenceManifest(
            result_sha256=RESULT_SHA256,
            protocol_sha256=PROTOCOL_SHA256,
            files=_collect(archive, protocol, groups, directory),
        )
        staged = directory / "references.zip"
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
            for item in manifest.files:
                info = ZipInfo(item.archive_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                output.writestr(
                    info,
                    (directory / item.path).read_bytes(),
                    compress_type=ZIP_DEFLATED,
                    compresslevel=9,
                )
        validate_reference_archive(staged, manifest)
        destination.hardlink_to(staged)
    return manifest
