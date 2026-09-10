# Colab CPU verification result

## Outcome

The first implementation learns the binary association task on the real larval
mushroom-body connectivity pattern. This run does **not** demonstrate delayed
memory learning, language acquisition, or a uniquely biological topology
advantage.

Each condition used 1500 training episodes and 512 balanced evaluation trials.
The numbers below are means across seeds 0, 1, and 2.

| Task | Control | Initial accuracy | Final accuracy | Final seed SD |
|---|---|---:|---:|---:|
| Association | Real, learning | 49.15% | 83.98% | 1.76 pp |
| Association | Rewired, learning | 49.35% | 72.85% | 16.69 pp |
| Association | Real, frozen | 49.15% | 49.15% | 2.51 pp |
| Delayed memory | Real, learning | 49.41% | 50.13% | 0.41 pp |
| Delayed memory | Rewired, learning | 50.52% | 50.00% | 0.00 pp |
| Delayed memory | Real, frozen | 49.41% | 49.41% | 1.92 pp |

Chance and a constant action both score 50%. Seed SD is descriptive uncertainty
across these three runs, not a confidence interval or significance test.

The rewired seed 1 loses the originally selected direct sensory-output edge:
its association accuracy is 53.71%, compared with 80.47% and 84.38% for the
other two rewired seeds. Consequently, the aggregate real-versus-rewired gap
is confounded by task accessibility. Do not interpret it as proof that fly
topology is intrinsically better for learning.

## Runtime and exact resumption

- Runtime: Colab CPU, Linux x86_64, two reported CPUs.
- Python: 3.13.15.
- Available RAM: 13,605,830,656 bytes.
- Data: 209 neurons, 7425 directed edges, left larval mushroom body.
- Verification wall time: **83.66 seconds**, excluding VM provisioning,
  dependency installation, transfer, and the later explicit test-output capture.
- Maximum observed kernel RSS: 162,948 KiB, including notebook/kernel overhead.
- Maximum observed child-process RSS: 105,220 KiB.

The verification first trained to 1000 episodes, resumed to 1500, and compared
against an independently initialized uninterrupted 1500-episode run. All
**18 checkpoint JSON files plus `summary.json` were byte-identical**.
This validates episode-boundary resumption within a matching runtime, not
cross-version or cross-architecture bitwise reproducibility.

The exact uploaded source archive had SHA-256:

```text
04c925e3d105735ce4fae00ed546df223ee4bdcd7436ab1397eab630c35fe137
```

That snapshot remains in `artifacts/flyrl-source.tar.gz`. The delivery also
contains this report, documentation, and a separate explicit test-output
capture script added after the experiment; the learning implementation is
unchanged.

## Evidence locations

- `results/colab/colab-resumed/summary.json`: individual and aggregate results.
- `results/colab/colab-resumed/*.json`: resumable experiment state.
- `results/colab/colab-uninterrupted/*.json`: independent reference state.
- `results/colab/colab-evidence.json`: runtime measurements and comparison count.
- `results/colab/*.log`: the three actual CLI invocation outputs.
- `artifacts/flyrl-results.zip`: original downloaded remote result archive.
- `artifacts/colab-tests.txt`: explicit remote test output.
- `data/provenance.json`: source version, checksums, and biological assumptions.

All **32 tests passed on Colab in 1.74 seconds** in the explicit output capture.
Ruff and a fresh full basedpyright invocation passed. Wheel and source
distribution builds passed. The shell runner passed `bash -n`; a Bash language
server was unavailable. The persistent Python language server retained stale
import diagnostics, while the fresh checker resolved the complete project
with zero errors or warnings.

## Next experimental decision

The reward-learning mechanism works on the association task. Before increasing
network size or adding natural text, the useful next experiment is a
task-accessibility-matched topology control and a short-delay memory sweep.
Those are proposals, not unfinished features of this first prototype.
