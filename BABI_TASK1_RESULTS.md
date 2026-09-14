# A real-connectome recurrent model partially learned bAbI Task 1

## Result

The fixed one-seed model answered **602 of 1,000 official test questions
exactly (60.2%)**.

This is **weak partial task learning**, not a solved result. The conventional
bAbI Task 1 threshold frozen before the run was 95%.

The completed scores and final recovery are valid, but the run was **not fully
protocol-conformant**: periodic recovery copies remained on the same Colab VM
instead of being copied outside the runtime. They would not have survived an
allocation loss during training. This deviation did not alter the completed
validation, ordinary-test, or ablation arithmetic.

The separate input-evidence check passed: accuracy fell from **60.2% to 15.7%**
when every preceding fact about the queried person was removed without
retraining. The 44.5-point drop exceeded the required 20 points, and the
removed-input score was below the fixed 27.2% ceiling derived from the strongest
answer-prior baseline.

The narrow conclusion is:

> This fixed-budget connectome-constrained model learned a useful but
> insufficient part of English bAbI Task 1, and its answers were materially
> sensitive to the supplied location evidence.

That does not establish general reasoning, general language understanding,
topology superiority, biological function, or performance on bAbI Task 2.

## Frozen benchmark

| Setting | Value |
| --- | --- |
| Task | English bAbI Task 1, single supporting fact |
| Recurrent core | Real Drosophila connectome graph |
| Nodes / directed edges | 16,384 / 1,187,999 |
| External RNN / attention | None / none |
| Topology comparison | None |
| Seed | 0 |
| Tokenizer | Unchanged TinyStories BPE-4096 |
| Training objective | Answer tokens and LF terminator only |
| Context / batch | 160 / 8 |
| Training budget | 12,000 updates; 96,000 sampled questions |
| Checkpoint selection | Validation exact count, then answer NLL, then earliest update |
| Primary metric | Unrestricted greedy, free-running exact answer |

The model was initialized from seed 0. No TinyStories model weights were loaded.
The externally fitted tokenizer was reused without adding or resizing tokens.
`hallway` spans two lexical tokens; loss was averaged within each question
before averaging the batch so it received the same question weight as the
single-token locations.

## Official data and split

The source was Meta's current ParlAI-hosted bAbI archive:

- archive SHA256:
  `f7f0bee187efca0d81c3daac1b162cda4eb7f9505dee5ad6846eabbed3dbf92e`;
- Task 1 train member SHA256:
  `749ea9f7c99070feb2d88c975a254417a0dcc8274add4435ae5ae24c7afc7e9d`;
- Task 1 test member SHA256:
  `55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b`;
- embedded CC BY 3.0 license SHA256:
  `d12f09a636365a040fd581ebab6bf018d6fe61973c6d4862f841a6aca2f53efa`.

The exact authenticated outer archive is retained at
[`artifacts/babi_tasks_1-20_v1-2-parlai.tar.gz`](artifacts/babi_tasks_1-20_v1-2-parlai.tar.gz).
Preparation therefore remains reproducible without relying on the mutable CDN
URL.

| Split | Questions | Exact unique prompts |
| --- | ---: | ---: |
| Train | 8,983 | 8,909 |
| Validation | 1,000 | 999 |
| Official test | 1,000 | 999 |

The official train file was split by whole episode with PCG64 seed 0. Seventeen
candidate-training questions that exactly matched validation prompts were
removed. The official test file did not influence that operation, training,
hyperparameters, checkpoint selection, or thresholds.

After selection, the untouched test was found to contain 16 exact prompts also
seen in retained train. Accuracy was 590/984 (59.96%) on train-novel test prompts
and 12/16 (75%) on the small seen subset. The official score remains all
1,000 questions.

## Validation and selection

| Update | Exact | Answer NLL |
| ---: | ---: | ---: |
| 0 | 0.0% | 8.3182 |
| 500 | 15.5% | 0.9090 |
| 2,000 | 15.9% | 0.8412 |
| 4,000 | 44.6% | 0.7075 |
| 8,000 | 58.7% | 0.5919 |
| **12,000** | **63.3%** | **0.5304** |

Update 12,000 was selected before the official test artifact was opened. The
selected checkpoint SHA256 is
`3c77d5b56f0aece9ffdfb6fe369d2ea6ec9969f889e35872c339e59c456b1a41`.

## Official test and baselines

| Method | Exact accuracy |
| --- | ---: |
| Uniform six-location chance | 16.67% |
| Train-majority answer | 14.9% |
| Train-fitted queried-person prior | 17.2% |
| Last supplied fact, ignoring queried person | 52.7% |
| **Connectome-constrained model** | **60.2%** |
| Symbolic latest-location oracle | 100% |

The model beat the strongest non-oracle heuristic by 7.5 points, but remained
34.8 points below the solved threshold. Its teacher-forced equal-question test
NLL was 0.5663 over 2,154 answer tokens.

### Accuracy by answer

| Gold answer | Correct | Accuracy |
| --- | ---: | ---: |
| bathroom | 91/149 | 61.1% |
| bedroom | 99/171 | 57.9% |
| garden | 118/187 | 63.1% |
| hallway | 96/154 | 62.3% |
| kitchen | 102/157 | 65.0% |
| office | 96/182 | 52.7% |

All 1,000 generations terminated within the eight-token limit and decoded to
exactly one of the six legal locations. The 398 errors were therefore wrong
location selections, not malformed, unterminated, or repaired outputs. The most
frequent individual confusions were `office -> kitchen` (27),
`bedroom -> office` (23), `office -> bedroom` (21), and
`bathroom -> garden` (21).

## Input-evidence check

For each test question, the fixed ablation removed all preceding facts about the
queried person while preserving other people's facts and the question:

| Condition | Exact | Answer NLL |
| --- | ---: | ---: |
| Ordinary test | 602/1,000 (60.2%) | 0.5663 |
| Queried-person facts removed | 157/1,000 (15.7%) | 1.1951 |

The generated answer changed on 541 questions. Both frozen gates passed:

```text
ordinary - removed = 44.5 points >= 20 points
removed = 15.7% <= 17.2% prior baseline + 10 points
```

This supports sensitivity to the supplied answer evidence. It does not identify
a reasoning algorithm or neural mechanism. Removing the evidence creates
unsupported questions and changes the input distribution.

## Runtime and recovery

The quality preflight passed before main training:

- disposable memorization: 12/12 after exactly 400 updates;
- five warmups and 20 timed updates;
- mean timed update: 0.2257 seconds;
- 1,000-question timing fixture: 15.42 seconds;
- optimizer-bearing checkpoint write: 0.223 seconds;
- peak allocated tensors: 442,086,400 bytes (421.6 MiB);
- conservative projection: 4,550.24 seconds, below the 5,400-second gate.

The one Colab Free Tesla T4 allocation existed from
2026-09-14 08:16:21 UTC to 09:07:07 UTC, 3,045.66 seconds including setup,
downloads, validation, and release. Torch was `2.11.0+cu128`, CUDA was 12.8,
and Python was 3.13.15. The server reported zero active sessions after release.

Recovery evidence includes:

- immutable checkpoint transactions at all validation milestones and the latest
  two scheduled saves;
- verified same-VM copies every 1,000 updates outside the run output tree;
- complete model, Adam, sampler RNG, sample ledger, exposure count, trace,
  validation history, and next-LR restoration;
- same-T4 state equality and reproduction of the first 32 ordinary and ablated
  outputs;
- a hash-chained `settings -> selection -> test opened -> completed` event log;
- strict local recomputation of checkpoint selection, all exact counts,
  baselines, score band, and both input-evidence inequalities.

The 162,682,050-byte result ZIP has SHA256
`15ebfd4a210ddf85cf6fa0623c5d89ebdc4944178811d45b4bb288f182d5be58`.
Strict local recovery printed `BABI_TASK1_RECOVERY_VERIFIED`.

### Recovery protocol deviation

The frozen protocol required every 1,000-update recovery bundle to be copied
outside the runtime. The implementation instead wrote
`/content/babi-task1/recovery/recovery-*` on the same Colab VM. These copies
protected against a torn run-output transaction but **not** against loss of the
allocation itself.

The final completed ZIP was downloaded, hash-checked, and strictly recovered
locally before the T4 was released. This establishes recovery of the completed
result, not mid-run off-runtime durability. The publication therefore does not
claim full operational protocol conformance and the experiment was not rerun.

## Published evidence

[`data/babi_task1_result/`](data/babi_task1_result/) contains:

- all 1,000 ordinary predictions and answer losses in
  [`ordinary.json`](data/babi_task1_result/ordinary.json);
- all 1,000 evidence-removed predictions and answer losses in
  [`removed.json`](data/babi_task1_result/removed.json);
- the selected checkpoint record, preflight, event chain, same-runtime
  restoration evidence, remote verification, local recovery report, and a
  SHA256 manifest;
- the explicit
  [`protocol-deviations.json`](data/babi_task1_result/protocol-deviations.json)
  record.

The full checkpoint ZIP is retained locally rather than committed. Its digest
and every compact machine artifact needed to audit the published claims are
recorded in
[`manifest.json`](data/babi_task1_result/manifest.json).

The frozen protocol is
[`BABI_TASK1_PROTOCOL.md`](BABI_TASK1_PROTOCOL.md), and the machine summary is
[`BABI_TASK1_RESULTS.json`](BABI_TASK1_RESULTS.json).

## Claim limits

- One seed and one fixed budget were run.
- bAbI is a small synthetic task; success does not imply broad language ability.
- The reused tokenizer was fitted on TinyStories text.
- The official test contains 16 exact train-seen prompts; the 984-prompt novel
  diagnostic reached essentially the same accuracy.
- The result does not compare connectome topology with another topology.
- Token-linked state is not attention, biological firing, localization, or
  causal importance.
- No bAbI Task 2 data or experiment was used.

## Attribution

- Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush,
  Bart van Merriënboer, Armand Joulin, and Tomas Mikolov,
  [*Towards AI-Complete Question Answering: A Set of Prerequisite Toy
  Tasks*](https://arxiv.org/abs/1502.05698v10).
- [Meta ParlAI bAbI build manifest at the pinned
  commit](https://github.com/facebookresearch/ParlAI/blob/ea366da91c93f2cec8fe29b6d93b37ed96ed45bd/parlai/tasks/babi/build.py#L13-L18).
- [Official current archive](https://dl.fbaipublicfiles.com/parlai/babi/babi.tar.gz).
- Dataset legal code:
  [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/legalcode).
