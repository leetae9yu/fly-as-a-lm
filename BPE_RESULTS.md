# BPE comparison: partial results

**Six of fifteen planned runs completed. Training is stopped.** The full seed-0
comparison and the seed-1 anatomical run finished and were recovered. The
seed-1 shuffled run was interrupted; its external checkpoint contains 4,250 of
8,000 updates. Eight other runs were not started. This is not a completed
three-seed study.

The anatomical model learned next-token prediction, but the parameter-matched
ordinary GRU and small Transformer were substantially better and faster in the
completed comparison. Free generation remained incoherent or repetitive.
These results do not establish a biological-wiring advantage.

## Protocol and scope

The budget was committed in [BPE_STUDY.json](BPE_STUDY.json), revision `548400e`,
before main training. [BPE_STUDY.md](BPE_STUDY.md) documents the full method.

- MaleCNS subset: 16,384 neurons and 1,187,999 directed edges, with strong
  optic-lobe selection bias. The full 166,700-neuron graph was not trained here.
- Train-only byte-level BPE: 4,096 tokens, fitted on all normalized WikiText-2
  training text. Train/validation/test contain 2,830,508 / 288,823 / 276,241 tokens.
- The test suffix starts at normalized character 196,608, excluding regions
  scored by earlier experiments.
- Each completed run: 8,000 updates, batch 8, context 32, 2,048,000 target tokens.
  Windows are sampled with replacement: approximately 0.724 nominal corpus
  passes, not an exhaustive epoch or training to convergence.
- Float32 AdamW, clipping 1, no weight decay or dropout. Predeclared learning
  rates were .003 for anatomical conditions and .001 for ordinary models.
- Every score uses the final target of 1,024 identical non-overlapping
  32-token-context windows in its split. This is **windowed BPE perplexity**,
  not standard full-document or word-level WikiText perplexity.
- Final-update reporting, with no early stopping or score-based choice of
  checkpoints, architectures, learning rates or training budgets.

The three fully trainable architectures are within 1% in parameter count.
The ordinary models learn embeddings; the anatomical model uses fixed bipolar
input codes. This is a practical model comparison, not an isolated manipulation
of topology. Real versus shuffled is the relevant wiring comparison.

## Completed seed-0 comparison

Lower NLL, bits/token and perplexity are better. Accuracy is next-token accuracy,
not next-word accuracy.

| Condition | Test NLL | Test bits/token | Test perplexity | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Anatomical, trained | 5.2313 | 7.5472 | 187.04 | 20.31% |
| Rewired, trained | 5.2826 | 7.6212 | 196.88 | 19.92% |
| Anatomical, frozen core | 5.9507 | 8.5851 | 384.02 | 13.57% |
| Ordinary GRU | **4.0348** | **5.8210** | **56.53** | **29.49%** |
| Small Transformer | 4.1792 | 6.0294 | 65.32 | 27.83% |

The anatomical model's test NLL fell from **8.3179 to 5.2313**, and its test
perplexity from **4,096.64 to 187.04**. It outperformed the frozen-core condition,
although that ablation intentionally has fewer trainable parameters.
The small real-versus-shuffled difference in this single paired seed is not
evidence of general anatomical superiority.

### Validation and measured training cost

| Condition | Trainable parameters | Validation perplexity | Training seconds | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| Anatomical, trained | 2,257,055 | 208.34 | 1,551.27 | 177.46 |
| Rewired, trained | 2,257,055 | 230.14 | 1,716.91 | 188.28 |
| Anatomical, frozen core | 1,052,672 | 425.00 | 218.59 | 132.79 |
| Ordinary GRU | 2,271,104 | **66.59** | **33.56** | 71.54 |
| Small Transformer | 2,248,096 | 67.55 | 74.94 | **63.34** |

Times are observed, synchronized training time, excluding initial/final
evaluation and checkpoint I/O. GPU memory is the main run's PyTorch peak
allocation, excluding driver/context memory and unused reservations. It is
different from the short disposable profile's peak.

The anatomical run took about **21 times** the Transformer's training time and
46 times the GRU's. This measures the current implementation, not an intrinsic
speed limit of RNNs or biological circuits. The ordinary GRU was faster than
the Transformer.

## Additional completed anatomical run

Seed 1 used the same budget and evaluation positions:

| Metric | Seed 1 anatomical model |
| --- | ---: |
| Train NLL | 5.6090 |
| Validation NLL / perplexity | 5.4045 / 222.39 |
| Test NLL / bits per token | 5.1523 / 7.4332 |
| Test perplexity / accuracy | 172.83 / 20.90% |
| Training seconds | 1,554.48 |
| Peak allocated GPU MiB | 177.46 |

Its test perplexity improved from 4,096.19 to 172.83. This reproduces learning
in another anatomical initialization, but the corresponding completed seed-1
controls are missing. We therefore do not publish a three-seed average,
uncertainty estimate or paired significance claim.

## Supplementary n-gram references

These fit counts on **all training tokens**, unlike the neural models'
sampled 2,048,000-target budget. They are not exposure-matched neural controls.
All use add-half smoothing and the same final test targets.

| Reference | Test NLL | Test perplexity | Test accuracy |
| --- | ---: | ---: | ---: |
| Unigram | 6.3735 | 586.10 | 5.18% |
| Bigram | 4.8875 | 132.62 | 21.00% |
| Trigram | 6.3709 | 584.59 | 29.69% |

The anatomical model did not beat the bigram's likelihood. The trigram's high
accuracy coexists with poor likelihood: add-half smoothing over 4,096 tokens
is a weak probability estimator for rare contexts. Accuracy alone is not
evidence of a better language model.

## Generation and core ablation

All models generate 128 tokens from the same training-only prompt using a
rolling 32-token context. There is no continuation from validation or test text.
Raw IDs, separate previews and jointly decoded prompt-plus-output text are
stored in each report.

Anatomical seed 0, excerpt from the greedy continuation:

```text
 , the <unk> of the <unk> , <unk> , <unk> , <unk> , <unk>
```

Anatomical seed 1, excerpt from the greedy continuation:

```text
 ) . the first time of the first of the series , and the firstth division divisionem to the southwest .
```

The other controls also exhibit repetition, malformed subwords or topic drift.
The GRU's greedy output repeats Japanese fragments following the Japanese
prompt fragment; the Transformer also falls into repeated `<unk>` phrases.
Better teacher-forced likelihood did not produce coherent free text here.
Upstream WikiText already contains literal `<unk>` tokens.

Zeroing the trained anatomical recurrence at evaluation raises seed-0 test NLL
to 9.5328 and seed-1 NLL to 9.5936. This destructive ablation changes the hidden
state distribution and shows dependence on the trained core; it does not by
itself identify a useful biological circuit.

## Bottleneck evidence: measured versus inferred

Existing T4 warm profiles measured 184.61 ms/update for the trainable anatomical
model, 25.54 ms for its frozen-core control, 4.99 ms for GRU and 9.82 ms for
Transformer. The large real/frozen gap points toward the trained-core path.

A separate **local CPU diagnostic**, not a T4 kernel profile, used a fresh
16,384-neuron model at batch 8/context 32, one CPU thread and Torch 2.14.0+cpu
on the arm64 workstation. It measured:

| CPU measurement | Time |
| --- | ---: |
| Unprofiled warm update, mean of 3 | 6.179 s |
| Forward/loss, synchronized mean of 2 | 1.857 s |
| Backward, synchronized mean of 2 | 4.398 s |
| Gradient clipping | 0.00185 s |
| Optimizer | 0.01035 s |

In one operator-profiled update, `aten::index` used 3.267 s of exclusive CPU
time across 1,376 calls; `aten::addmm` used 2.192 s across 64 calls.
The custom backward contains 19 edge-gradient chunks per time step, or 608
chunks per update. Repeated gathers, sparse multiplication, value reordering
and dispatch are candidates for optimization.

The CPU operator percentages are **not GPU percentages**. A detailed GPU kernel
trace was not obtained before the runtime interruption and subsequent stop.
No kernel optimization was applied or benchmarked in the main experiment.
The retained [CPU operator summary](results/bpe-bottleneck-cpu/real.json)
separates CPU and device event accounting.

## Interruption, recovery and publication status

The T4 assignment disappeared during seed-1 shuffled training. Replacement
requests returned `Service Unavailable`; the user identified this as the GPU
usage limit. The API messages alone do not establish the termination cause.
The user then explicitly stopped further experiments and automatic retries.

| Seed | Real | Shuffled | Frozen | GRU | Transformer |
| --- | --- | --- | --- | --- | --- |
| 0 | Complete | Complete | Complete | Complete | Complete |
| 1 | Complete | Interrupted: 4,250-update external backup | Not run | Not run | Not run |
| 2 | Not run | Not run | Not run | Not run | Not run |

All six completed source checkpoints were recovered. Their separate
8,000-to-8,004 continuation checks reproduced the stored generation and restored
parameters, Adam moments, window RNG, update count and trace exactly.
Subsequent CUDA continuation differed by at most **2.3842e-7** in parameters and
**3.0268e-9** in Adam tensors; recorded continuation losses and window RNG agreed.
GRU and Transformer continuation were bitwise exact in these checks.
Source checkpoint hashes matched the recovery evidence.

The interrupted checkpoint and its original initial scores were also downloaded
and integrity-checked. It is not a completed result and has no final score in
this report. If resumed in a future authorized study, its reported
`training_seconds_this_session` would exclude the lost runtime's earlier work.
See [the interruption record](results/bpe-runtime-interruption.json).

### Published evidence

| Run | Final report | Restoration and continuation |
| --- | --- | --- |
| Seed 0 real | [report](results/bpe-main/seed-0/real/report.json) | [verification](results/bpe-resume/seed-0/real/verification.json) |
| Seed 0 shuffled | [report](results/bpe-main/seed-0/shuffled/report.json) | [verification](results/bpe-resume/seed-0/shuffled/verification.json) |
| Seed 0 frozen | [report](results/bpe-main/seed-0/frozen/report.json) | [verification](results/bpe-resume/seed-0/frozen/verification.json) |
| Seed 0 GRU | [report](results/bpe-main/seed-0/gru/report.json) | [verification](results/bpe-resume/seed-0/gru/verification.json) |
| Seed 0 Transformer | [report](results/bpe-main/seed-0/transformer/report.json) | [verification](results/bpe-resume/seed-0/transformer/verification.json) |
| Seed 1 real | [report](results/bpe-main/seed-1/real/report.json) | [verification](results/bpe-resume/seed-1/real/verification.json) |

Model weights, recovery ZIPs and the raw diagnostic trace remain local, not in
Git. The committed reports include the full training traces and generated IDs.
The corpus, tokenizer, fixed plan, source code and numerical evidence are public.

Recorded software checks were 192 passing tests plus one CUDA-only skip locally,
48 passing model tests on T4, and three subsequent study/CLI checks.
Static checks and package builds passed during implementation. This publication
audits saved artifacts and updates documentation; it runs no further training,
profiling or training-containing test suite.

## Interpretation

This result supports trainability of the constrained computational model.
It does not support competitive efficiency, fluent language, anatomical
superiority, or a claim that living flies learn human language. Limited data
exposure, one complete paired seed, different input representations, untuned
model-specific learning rates and the biased subgraph limit generalization.
Longer pretraining, input-code learning and sparse-kernel work are possible
future investigations, not results of this stopped study. SFT and GRPO were
discussed but not implemented or run.
