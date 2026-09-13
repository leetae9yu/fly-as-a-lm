# Frozen regional next-token probe experiment

This prospective diagnostic asks where next-token information is linearly
accessible inside the trained N5,600 central connectome. It follows the
anatomy-port factorial, where forcing the language readout through all 97 MBONs
was worse than capacity-matched random readout neurons in every seed.

The experiment does not retrain the recurrent connectome. It freezes the
existing models and trains equal-capacity linear decoders from predeclared
97-neuron groups.

## Frozen source models

- Graph: `malecns_v1_cb_intrinsic_n5600.npz`
- Graph SHA256:
  `738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935`
- Graph size: 5,600 neurons and 1,187,928 directed pairs
- Corpus SHA256:
  `c750d2aabe2311cb713d3afed98c6fa8b1f044f092a7727a3a8cef278ec22b5e`
- Corpus fingerprint:
  `fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575`
- Anatomy-array SHA256:
  `e9e230da2dbf8fa3c6bf2e10df571dc5519203518d196b3956e19a83429e4a78`
- Regional-group manifest SHA256:
  `fe77d4bd06ed502107149d5e39b9fe992ef5e0c87a5e1cc7ec7bdf0cb0db5d75`
- Source-manifest SHA256:
  `5a41a67f5570669c7d038f28f45cf7759de94210dff9857cb64bcc7654229f36`

The twelve source checkpoints are the final `real-random_random` and
`shuffled-random_random` models for each seed 7-12. Every source archive,
checkpoint, report, runtime, AdamW state, RNG state and learned-parameter
fingerprint is fixed before probe fitting.

All recurrent parameters, sensory codes and original output parameters remain
frozen. Shuffled checkpoints retain their original seed-specific shuffled
endpoints. A new probe is not a checkpoint continuation and never changes the
source model.

## Equal-capacity groups

Every probe receives exactly 97 signed neuron states and has one affine
97-to-4,096 softmax head: 401,408 trainable values.

For each source seed, the same exact groups are reused across real and shuffled
wiring:

- five seeded 97-of-313 ALPN subsets;
- five seeded 97-neuron Kenyon-cell subsets after excluding source sensory and
  original readout membership;
- all 97 MBONs;
- 97 unannotated internal neurons greedily matched to MBON in/out degree and
  weighted in/out strength;
- the 97 highest weighted-degree unannotated internal neurons; and
- the source model's exact 97 trained random readout neurons.

This makes 14 heads per checkpoint and 168 heads overall. ALPN and Kenyon
subsets use
`default_rng(SeedSequence([20260913, source_seed, group_id, draw]))`.
Subsets are averaged within a group; they are not experimental replicates and
are never selected by activity or held-out performance.

Degree-matched and centrality controls exclude every ALPN, Kenyon cell, MBON,
source sensory neuron and trained readout neuron. Exact graph indices and
stable body IDs are frozen in
`data/central_connectome/regional_probe_groups.json`.

## Causal state extraction

Teacher forcing consumes token `x_t`, records the selected components of the
resulting recurrent state, and pairs them with target `x_{t+1}`. State resets
only between stories and is carried across 64-token processing chunks.
Batch-padding positions, first-token targets and cross-story transitions are
excluded.

All within-story pairs are used:

| Split | Stories | Targets |
| --- | ---: | ---: |
| train | 1,000 | 229,745 |
| validation | 16 | 4,082 |
| test | 16 | 3,139 |

Extraction uses eight independent stories per GPU batch. The union of all
selected groups is extracted once per checkpoint under `no_grad`; only selected
states are retained. Source parameters, buffers, model mode, progress and RNG
must be unchanged afterward.

## Probe fitting

Each neuron is standardized by its training-only population mean and standard
deviation, floored at .001. Constant neurons remain in place.

- weights initialized to zero;
- bias initialized to log add-half-smoothed training unigram probabilities;
- AdamW with weight decay 0;
- explicit weight-only L2 coefficient .0001;
- gradient clipping 1;
- 750 updates;
- batch 2,048, sampled with replacement by a private CPU generator;
- generator seed `20260913 + source_seed`, restarted identically for every
  head;
- cosine learning rate from .01 toward .0001; and
- float32 fitting on CUDA.

There is no hyperparameter search, early stopping, checkpoint selection,
validation refit or budget extension. Validation and test scores are evaluated
once after update 750. Training-only add-half unigram and within-story bigram
references are also recorded.

## Prospective decision

The positive-control validity difference for seed `s` is:

```text
V_s = unigram_test_NLL - trained_readout_probe_test_NLL
```

The diagnostic is valid only if all six `V_s` values are positive and their
median is at least .10 nat/token.

Candidate groups are ALPN, Kenyon and centrality. For each candidate `g`:

```text
A_s,g = degree_matched_test_NLL - candidate_test_NLL
Q_s,g = unigram_test_NLL - candidate_test_NLL
```

Positive `A` means that the candidate beats an equally wide structural
comparator; positive `Q` means it beats frequency-only prediction. Ties count
against success.

For each six-seed vector, a one-sided exact sign probability is computed. The
candidate intersection-union probability is the larger of its `A` and `Q`
probabilities. Holm correction is applied across the three fixed candidates at
familywise alpha .05.

A candidate is nominated only when:

- the positive-control validity gate passes;
- all six `A` and all six `Q` values are strictly positive;
- both median improvements are at least .10 nat/token; and
- its intersection probability survives Holm correction.

No candidate, one candidate and multiple candidates produce
`no_candidate`, `single_candidate` and `multiple_candidates`. A failed
positive control produces `probe_invalid`.

MBON-versus-degree-matched deficits, wiring differences, real-vs-shuffled
localization interactions, original-head comparisons, subset dispersion,
per-neuron training variance and the fraction of states with absolute value at
least .99 are descriptive only.

## Sealing and recovery

No validation or test metric is printed during fitting. Metric-free
source-completion markers may be observed and individually downloaded. The
combined summary is released only after all 168 heads finish.

Local recovery must verify the frozen input protocol, all twelve source archive
hashes, exact group indices, complete traces, finite float32 parameter arrays,
feature hashes and labels, Tesla T4 runtime, source immutability and the
independently recomputed aggregate decision.

## Interpretation limits

This measures linear accessibility from 97 neurons under models trained with
random input and readout ports. It does not establish biological function,
causal necessity or a unique language circuit. The test stories and seeds were
seen in earlier experiment analysis, so this is a prospectively fixed
diagnostic rather than independent confirmation. A negative subset result
cannot exclude sparse, distributed or nonlinear information elsewhere in a
complete region.

## Measured result

The fixed experiment completed on 2026-09-13. All twelve source checkpoints and
all 168 heads finished before the aggregate summary was released. Strict local
recovery reproduced the remote decision exactly.

The prospective verdict is:

```text
single_candidate: alpn
```

### Decision gates

The trained-readout positive control beat the unigram baseline in all six real
seeds. Its six NLL improvements were `.9887`, `.9430`, `.7040`, `1.0286`,
`1.0841` and `.8312` nat/token, with median `.9659`. The diagnostic validity
gate therefore passed.

| Candidate | Matched wins | Median matched gain | Unigram wins | Median unigram gain | Intersection p | Holm rejected | Nominated |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ALPN | 6/6 | .1532 | 6/6 | .4388 | .015625 | yes | yes |
| Kenyon | 2/6 | -.1410 | 6/6 | .0779 | .890625 | no | no |
| Centrality | 0/6 | -.1851 | 6/6 | .0463 | 1.0 | no | no |

ALPN's six degree-matched gains were `.4594`, `.1857`, `.1685`, `.0044`,
`.1378` and `.0668` nat/token. Its six unigram gains were `.5700`, `.2484`,
`.5157`, `.4062`, `.2590` and `.4713`. Both one-sided exact sign probabilities
were `1/64`; the resulting intersection probability `.015625` passed the first
Holm threshold `.05 / 3`.

Kenyon and centrality both beat the unigram reference in all six real seeds, but
neither beat the equal-width degree-matched comparator consistently or reached
both practical-effect thresholds. They were not nominated.

MBON probe NLL was worse than degree-matched NLL in all six real seeds. The
deficits were `.0605`, `.0119`, `.1925`, `.3339`, `.0600` and `.2654`
nat/token, with median `.1265`. This agrees with the earlier observation that
forcing the trained output through MBONs was harmful.

### Absolute held-out performance

The table reports means over six source seeds after averaging the five fixed
draws for ALPN and Kenyon within each seed.

| Group | Real NLL | Real PPL | Real acc. | Shuffled NLL | Shuffled PPL | Shuffled acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alpn` | 5.5924 | 270.98 | 10.87% | 5.5266 | 254.34 | 11.61% |
| `kenyon` | 5.9146 | 370.80 | 8.31% | 5.9339 | 377.85 | 8.24% |
| `mbon` | 5.9168 | 371.57 | 8.22% | 5.9163 | 372.13 | 8.39% |
| `degree_matched` | 5.7628 | 321.61 | 9.62% | 5.8719 | 357.11 | 8.63% |
| `centrality` | 5.9576 | 386.68 | 8.02% | 5.9572 | 386.52 | 7.99% |
| `trained_readout` | 5.0742 | 161.19 | 17.04% | 5.0103 | 152.18 | 16.50% |

The corpus-derived unigram reference was NLL `6.0041`, PPL `405.10` and
accuracy `7.14%`. The descriptive within-story bigram reference was NLL
`5.2508`, PPL `190.73` and accuracy `25.23%`. Local recovery recomputed both
references directly from the frozen corpus and matched every stored source
record exactly.

### What the ALPN result does and does not mean

The real-model ALPN result establishes that next-token information is linearly
accessible from the fixed ALPN subsets more strongly than from the predeclared
degree-matched unannotated neurons. It is the only candidate that satisfies the
prospective cross-seed, practical-effect and familywise statistical gates.

It does not show that intact biological ALPN wiring causes that accessibility.
The descriptive real-minus-shuffled ALPN NLL differences were `.0397`, `.0595`,
`-.1768`, `.2052`, `.3430` and `-.0761` nat/token. The median was `.0496`, with
real wiring better in only two of six seeds. A real-versus-shuffled interaction
was not part of the nomination gate.

ALPN states also had more usable dynamic range. Across real seeds, the median
training-state saturation fraction was `64.9%` for ALPN, compared with `95.3%`
for Kenyon, `95.9%` for MBON, `93.4%` for degree-matched neurons and `96.8%`
for centrality neurons. Median per-neuron training variance was `.00674` for
ALPN versus `.00464` for MBON. These are descriptive mechanisms, not separate
selection criteria.

### Execution and recovery evidence

- Final sealed input:
  `artifacts/regional-probe-t4-input.zip`
  (`3a691231a0ca173ea78d1dcc0e36b299e8716dff76fc478650d87bd24e602dc0`,
  16,562,517 bytes).
- Final sealed results:
  `artifacts/regional-probe-t4-results.zip`
  (`ac7434b9f4ae313e48005569ab18a659a930c731e422fe96c09bc748a86a73ca`,
  435,276,937 bytes).
- Local recovery:
  `results/regional-probe-t4/recovery.json`
  (`f31e1c8f5835270f10116cb29051ebe02aa92a15a003efc5b86332059b23b229`).
- Every source archive passed CRC validation when downloaded. All twelve
  partial archives are byte-identical to their members in the combined result
  ZIP.
- Recovery verified all 12 source identities, 168 group identities, exact
  held-out labels and feature hashes, float32 parameter arrays, complete
  750-update traces, T4 runtime evidence, source immutability and remote/local
  aggregate equality. It emitted `REGIONAL_PROBE_RECOVERY_VERIFIED`.
- Source work took 1,019.5 seconds in total, median 83.87 seconds per source.
  Maximum recorded CUDA allocation was 581,079,040 bytes. The runtime was a
  Tesla T4 with Torch `2.11.0+cu128`, CUDA 12.8, one CPU thread, TF32 disabled
  and deterministic-algorithm mode disabled, matching the source checkpoints.

The first launch stopped before completing any source because bare device
`cuda` was passed to `torch.cuda.set_device`. The recovery fix resolves a bare
CUDA device to the current concrete index. A regression test was added, the
source distribution and sealed input were rebuilt, and standalone setup
revalidated every input before the successful run. No held-out result from the
failed launch existed or was inspected.

Local recovery independently recomputes the complete prospective decision from
the hash-validated per-head records. It validates saved parameter hashes,
shapes, dtypes and finiteness, but does not rerun all 168 held-out matrix
evaluations from saved weights; this remains a verification limitation.
