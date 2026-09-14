"""One sealed source extraction, fourteen fits and independently recoverable ZIP."""

import gc
import platform
import time
from pathlib import Path

import numpy as np
import torch

from flyrl.regional_probe import extract_features, feature_stats, fit_probe
from flyrl.regional_probe_types import FeatureCache
from flyrl.story_data import StoryCorpus
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import ProbeRuntime, SourceArtifact
from scripts.regional_probe_artifacts import (
    load_heldout,
    save_head,
    save_heldout,
    seal_zip,
)
from scripts.regional_probe_protocol import ProbeSourceCheckpoint
from scripts.regional_probe_schema import ProbeBaselines, ProbeResult, ProbeScore
from scripts.regional_probe_worker_support import (
    WorkerInputs,
    parameter_fingerprint,
    restore_source,
    source_files,
    union_indices,
)


def slice_cache(
    cache: FeatureCache, union: tuple[int, ...], indices: tuple[int, ...]
) -> FeatureCache:
    """Select manifest columns in their exact declared order, with C-order bytes."""
    columns = tuple(union.index(index) for index in indices)
    return FeatureCache(np.ascontiguousarray(cache.features[:, columns]), cache.labels)


def run_source(
    root: Path,
    inputs: WorkerInputs,
    stories: StoryCorpus,
    source: ProbeSourceCheckpoint,
    baseline_scores: tuple[ProbeScore, ProbeScore],
) -> tuple[tuple[ProbeResult, ...], ProbeBaselines]:
    """Freeze one exact source, extract three unions once, and finish fourteen heads."""
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    directory = root / "sources" / Path(source.filename).stem
    learner, report = restore_source(directory, source, inputs)
    parameter_count = sum(parameter.numel() for parameter in learner.model.parameters())
    before_buffers = tuple(
        buffer.detach().clone() for buffer in learner.model.buffers()
    )
    plan = next(plan for plan in inputs.groups.seeds if plan.seed == source.seed)
    union = union_indices(plan)
    caches = tuple(
        extract_features(
            learner.model, tokens, split, union, config=inputs.protocol.extraction
        )
        for tokens, split in zip(stories.streams, stories.metadata.splits, strict=True)
    )
    train, valid, test = caches
    for cache in caches:
        _ = feature_stats(cache)
    output = root / "results" / f"seed-{source.seed}-{source.wiring}"
    output.mkdir(parents=True, exist_ok=False)
    heldout_path = output / "heldout.npz"
    heldout_hash = save_heldout(heldout_path, union, valid, test)
    restored_valid, restored_test = load_heldout(heldout_path, union, heldout_hash)
    results: list[ProbeResult] = []
    head_paths: list[Path] = []
    probe_config = inputs.protocol.probe.model_copy(
        update={"seed": inputs.protocol.probe.seed + source.seed}
    )
    for group in plan.groups:
        sliced_train, sliced_valid, sliced_test = (
            slice_cache(cache, union, group.indices) for cache in (train, valid, test)
        )
        result = fit_probe(
            sliced_train,
            sliced_valid,
            sliced_test,
            probe_config,
        )
        for restored, stats in zip(
            (restored_valid, restored_test), result.features[1:], strict=True
        ):
            if feature_stats(slice_cache(restored, union, group.indices)) != stats:
                message = "Stored heldout group slice differs from fitted features"
                raise ValueError(message)
        path = output / f"{group.name}-{group.draw}.json"
        _ = save_head(path, result, source, group, inputs.protocol.extraction)
        head_paths.append(path)
        results.append(
            ProbeResult(
                seed=source.seed,
                wiring=source.wiring,
                group=group.name,
                draw=group.draw,
                valid=ProbeScore.model_validate(result.valid.model_dump()),
                test=ProbeScore.model_validate(result.test.model_dump()),
            )
        )
    if (
        parameter_fingerprint(learner) != source.parameter_fingerprint
        or any(
            not torch.equal(before, after)
            for before, after in zip(
                before_buffers, learner.model.buffers(), strict=True
            )
        )
        or learner.updates != report.updates
        or tuple(learner.trace) != report.trace
        or len(results) != inputs.protocol.heads_per_checkpoint
    ):
        message = "Regional probe mutated its source or left incomplete heads"
        raise ValueError(message)
    source_files(directory, source)
    baselines = ProbeBaselines(
        seed=source.seed,
        wiring=source.wiring,
        unigram=baseline_scores[0],
        bigram=baseline_scores[1],
        original_head_nll=report.final.test.nll,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cuda = learner.model.weight.is_cuda
    runtime = ProbeRuntime(
        gpu=torch.cuda.get_device_name(learner.model.weight.device) if cuda else "CPU",
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=str(learner.model.weight.device),
        seconds=time.perf_counter() - started,
        peak_allocated_bytes=(
            torch.cuda.max_memory_allocated(learner.model.weight.device) if cuda else 0
        ),
        threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32 is True,
    )
    artifact = SourceArtifact(
        source=source,
        protocol_sha256=file_digest(root / "flyrl-0.1.0/protocol.json"),
        source_parameter_count=parameter_count,
        runtime=runtime,
        union_indices=union,
        heldout_file=heldout_path.name,
        heldout_sha256=heldout_hash,
        heads=tuple(path.name for path in head_paths),
        baselines=baselines,
    )
    evidence_path = output / "source.json"
    _ = evidence_path.write_text(artifact.model_dump_json() + "\n")
    files = (
        evidence_path,
        heldout_path,
        *(path for head in head_paths for path in (head, head.with_suffix(".npz"))),
    )
    if set(output.iterdir()) != set(files):
        message = "Regional probe source output membership differs"
        raise ValueError(message)
    seal_zip(root / f"seed-{source.seed}-{source.wiring}-probes.zip", files)
    del learner, caches, train, valid, test, restored_valid, restored_test
    _ = gc.collect()
    torch.cuda.empty_cache()
    return tuple(results), baselines
