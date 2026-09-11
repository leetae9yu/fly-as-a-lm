"""The committed study fixes data, budgets and paired model settings."""

from pathlib import Path

from scripts import run_bpe_study


def test_study_jobs_share_budget_and_window_stream() -> None:
    # Given: the committed plan, before any held-out model selection.
    plan = run_bpe_study.Study.model_validate_json(Path("BPE_STUDY.json").read_bytes())
    # When: enumerating its initial comparison and independent seed repetitions.
    jobs = plan.jobs()
    # Then: all five conditions get all seeds and the same training token budget.
    assert len(jobs) == 15
    assert {job.seed for job in jobs} == {0, 1, 2}
    assert {job.condition for job in jobs} == {
        "real",
        "shuffled",
        "frozen",
        "gru",
        "transformer",
    }
    assert {job.batch_size * job.context * plan.updates for job in jobs} == {2_048_000}
    assert all(job.generation_context == "windowed" for job in jobs)
    assert len({(job.seed, job.condition) for job in jobs}) == 15
