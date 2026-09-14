# Frozen ALPN specificity and causal-intervention experiment

This prospective experiment follows the regional probe's ALPN nomination. It
asks whether that result survives ALPN-specific structural and activity
comparators on unscored text, and whether ALPN outgoing state deviations
contribute selectively to the original frozen decoder.

No recurrent parameter, sensory code, original readout, source checkpoint or
previous probe is retrained. Fresh losses remain sealed until every required
source and arm completes.

## Frozen sources

- Graph:
  `data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz`
- Graph SHA256:
  `738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935`
- Training corpus SHA256:
  `c750d2aabe2311cb713d3afed98c6fa8b1f044f092a7727a3a8cef278ec22b5e`
- Training-corpus fingerprint:
  `fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575`
- Tokenizer SHA256:
  `9dbe72484ae01b374801f3f7f8ffdcba3f126cb71774dc5f36ef75d06d064e27`
- Anatomy-array SHA256:
  `e9e230da2dbf8fa3c6bf2e10df571dc5519203518d196b3956e19a83429e4a78`
- Regional-group SHA256:
  `fe77d4bd06ed502107149d5e39b9fe992ef5e0c87a5e1cc7ec7bdf0cb0db5d75`
- Regional result SHA256:
  `ac7434b9f4ae313e48005569ab18a659a930c731e422fe96c09bc748a86a73ca`
- Source-checkpoint protocol SHA256:
  `6b134d0e928caacf13900ccf793a65954e02dadef6e40eee014fa7feb9bb0dba`

The twelve source models are the exact final `real-random_random` and
`shuffled-random_random` checkpoints for seeds 7-12. The five existing
97-neuron ALPN probes and one trained-readout probe per source are reused
without refitting. Their 144 JSON/NPZ files total 103,987,229 bytes; the
ordered path-plus-file-hash digest is
`0d0ac287ebc406284ace0c5ce2467105541a48e7d94907b289c8d9b28c210f17`.

## Fresh evaluation boundary

The existing downloaded source prefix contains 1,100 complete unique stories.
Exactly 1,032 entered the previous train, validation and test splits. This
experiment uses all remaining 68 stories in original source order.

- Consumed source bytes: 1,027,216
- Consumed-prefix SHA256:
  `b888ac8b6858ea8ce4547b19da58740bb364df28b7f28778b73f9121a9137fa6`
- Fresh stories: 68, all unique and disjoint from the prior 1,032 identities
- Encoded tokens: 12,310
- Within-story next-token targets: 12,242
- Token-array SHA256:
  `62083864627b36147b6c16d3326f94fa6ccc63f5acc2c384e018b4c73891920f`
- Offset-array SHA256:
  `f8b75050de6867464d8464f32618ea99652f3621d4d4a1afd9618465e2391784`
- Ordered story-identity SHA256:
  `d87d6abe6ac9c5dfa66f3e34e73a5737225d75ad8f239519ebae68d4277bb96d`

Text normalization and the 4,096-token tokenizer are unchanged. A story is
rejected if its case-folded whitespace five-word-set Jaccard similarity to any
previous story is at least .8. The measured maximum was .0722. Selection,
encoding, hashes and the rejection rule are frozen before any fresh score.

These texts are newly scored relative to recorded project artifacts. They are
not a representative random sample, new model seeds or proof of no exposure
outside the recorded experiment.

## ALPN-specific matched controls

The target groups are the five existing 97-of-313 ALPN draws for each seed.
Subsets may overlap and are averaged within seed; they are not independent
replicates.

Eligible controls exclude every ALPN, Kenyon cell, MBON, source sensory neuron
and trained readout neuron. Exact eligibility is seed-specific and shared
between its real and shuffled models.

Each ALPN draw receives two 97-distinct-neuron controls:

- `S`: structure and learned propagation matched;
- `M`: the same coordinates plus training-state activity matched.

The joint real/shuffled matching coordinates are:

1. original-graph `log1p` in-degree and out-degree;
2. original-graph `log1p` incoming and outgoing absolute edge strength;
3. for each wiring, `log1p` learned outgoing-weight column L2 norm;
4. for each wiring, directed distance from source sensory neurons;
5. for each wiring, directed distance to trained readout neurons;
6. for `M`, each wiring's signed training-state mean;
7. for `M`, each wiring's `log(max(population_std, .001))`; and
8. for `M`, each wiring's fraction of training states with `abs(h) >= .99`.

Distances are capped at three, including unreachable nodes. Activity uses all
229,745 old training-story targets and may not consume fresh stories.
Coordinates are standardized over the ALPN and eligible pool for that seed;
zero scale becomes one.

Each draw uses a float64 rectangular minimum-cost assignment without
replacement. Rows and candidates are ordered by stable body identity and ties
retain that order. The same realized controls are reused under real and
shuffled wiring. Controls may overlap across draws and between `S` and `M`;
overlap is reported.

Before fresh scoring, every promised coordinate in every realized control must
have absolute standardized mean difference at most .10 and empirical two-sample
KS distance at most .20. Any failure produces
`insufficient_common_support`. Calipers, targets, pool and matcher are not
relaxed after this check.

## Probe confirmation

Existing ALPN and trained-readout probes retain their saved weights,
normalization and training-unigram bias. Only the ten new `S`/`M` probes per
checkpoint are fitted: 120 new heads total.

Each new probe keeps the previous fixed 97-to-4,096 configuration:

- 401,408 parameters, float32 CUDA;
- zero weight initialization and add-half training-unigram bias;
- AdamW weight decay zero and explicit weight-only L2 `.0001`;
- clipping one, 750 updates and batch 2,048 with replacement;
- private CPU sampler seed `20260913 + source_seed`, restarted per head;
- cosine learning rate `.01` toward `.0001`; and
- training-only mean/std with floor `.001`.

There is no tuning, validation selection, early stopping, refit or budget
extension. Previous test scores are not treated as fresh evidence.

## Centered outgoing-signal intervention

Let `h_(t-1)` be the intervention trajectory and `d_t` the ordinary sensory
drive. Baseline recurrence uses:

```text
u_t = h_(t-1) + d_t
h_t = (1-leak) h_(t-1) + leak*tanh(W u_t + b + d_t)
```

For group `G`, intervention replaces only its recipient-dependent outgoing
signal with the source-specific old-training mean `mu_G`:

```text
u_t[G] = mu_G + d_t[G]
```

The local state update and leak still use the unmodified `h_(t-1)`. The initial
outgoing state remains zero on the first token; replacement starts on the
second token and continues across chunk boundaries. All self-loops and group
edges remain. No state, edge, bias, sensory code or readout parameter is
zeroed or refitted.

The original trained readout scores every resulting state. Each checkpoint has
baseline plus fifteen intervention arms: five ALPN, five `S` and five `M`.
Teacher forcing covers all 12,242 targets, resets only between stories and
never scores padding or cross-story transitions.

An empty intervention must reproduce baseline. A readout-state mean-replacement
positive control must worsen original-head NLL in all six seeds of both
wirings, with median damage at least .01.

Sixty-four target positions are selected before scoring with
`default_rng(20260914)` from the complete target range, without replacement and
in sorted order. They form the fixed dense-replay audit subset for every source,
probe and intervention.

## Prospective contrasts

For seed `s`, wiring `w` and comparator `c` in `{S, M}`:

```text
A_s,w,c = matched_probe_NLL - ALPN_probe_NLL
Q_s,w   = unigram_NLL - ALPN_probe_NLL

C_s,w,g = intervened_original_NLL(g) - intact_original_NLL
D_s,w,c = C_s,w,ALPN - C_s,w,c
K_s,c   = D_s,real,c - D_s,shuffled,c
```

Five draws are averaged within source before the six seeds are summarized.
NLL is token-weighted over all targets; perplexities are never averaged.

Assay validity additionally requires the original head and reused
trained-readout probe each to beat the training unigram in every seed of both
wirings, with median gain at least .10. Failure yields `assay_invalid`.

The fixed-sequence family at alpha .05 is:

| Gate | Required real-wiring components | Seedwise practical requirement |
| --- | --- | --- |
| Accessibility | `A_S`, `A_M`, `Q` | all six positive; each median >= .10 |
| Causal specificity | `C_ALPN`, `D_S`, `D_M` | all six positive; each median >= .01 |
| Wiring moderation | `K_S`, `K_M` | all six positive; each median >= .01 |

For fresh-text inference, each contrast is first computed per story, then draws
and the six seeds are averaged within story. A one-sided exact sign probability
uses the 68 story values; ties count against success. A gate's
intersection-union probability is the largest component probability. The next
gate is inferential only if the preceding gate passes.

The seed unanimity rules are robustness requirements because ALPN was selected
using these seeds. Story signs concern the fixed checkpoint ensemble and do not
create 408 independent replicates. The `.01` causal thresholds are new
approximately one-percent perplexity conventions, not power-derived limits.

Final verdicts are:

- `insufficient_common_support` before fresh scoring when matching fails;
- `assay_invalid` when validity controls fail;
- `joint_success` when all three ordered gates pass;
- `accessibility_causal` when the first two pass but wiring moderation fails;
- `accessibility_only` when accessibility passes but causal specificity fails;
- `no_accessibility_specificity` when the first gate fails.

Later estimates remain descriptive when the fixed sequence stops.

## Sealing and recovery

Calibration consumes only old training data and emits metric-free balance
status. The evaluation seal fixes realized controls, balance evidence, training
means, reused and fitted probe hashes, fresh corpus and worker before scoring.

No fresh loss, accuracy, contrast or gate result is printed until every
required source and arm completes. Only metric-free source-completion markers
may be downloaded. Retries retain identical inputs and never replace a seed.

Artifacts retain per-target loss and prediction evidence for every probe and
intervention, the 97 original-readout states for baseline and interventions,
and fresh selected states for every probe. Local recovery independently
recomputes all per-story and aggregate metrics from the per-target evidence.
It additionally recomputes dense logits from saved states and parameters at
64 target positions sampled without replacement by
`default_rng(20260914)` and sorted before use. The audit-position int64-array
SHA256 is
`0c665e8006a2b1d60edcf159d52b2d9f8f5fd2b968abdb8326519343bb0fabd4`.
Recovery then reproduces all contrasts, story signs, gates and the remote
verdict. Full dense replay of all 12,242 targets is not required. Recovery also
verifies sources, model
parameters and buffers, ports, topology, progress, RNG, matching, training
means, traces, dtypes, shapes, runtime flags and source immutability.

## Result status

The common-support gate completed on 2026-09-13 with the terminal verdict
`insufficient_common_support`. This is a matching failure, not a negative
accessibility or causal result.

The sealed calibration input was
`artifacts/alpn-causal-calibration-input.zip` (SHA256
`8376344dcf168ba50a0b151812dcb9c22518d0c61ca75cb70315439914ca3d32`).
It authenticated 143 project files and all twelve source archives. Two local
builds were byte-identical. The result is
`artifacts/alpn-causal-calibration-results.zip` (SHA256
`a7b4193be2e26ce95e4f7ca831e0daf3b0a231d7d0bccc2cf10c54955df2b5ef`).
Its sole `calibration.json` member passed CRC validation.

The runtime was a verified Tesla T4 with Python 3.13.15, Torch
2.11.0+cu128 and CUDA 12.8. A real CUDA matrix multiplication passed before
the run. All six seeds completed before the aggregate status was exposed.

Every one of the five `S` and five `M` controls failed at least one promised
balance bound in every seed:

| Seed | `S` failed | `S` max SMD / KS | `M` failed | `M` max SMD / KS |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 5/5 | .4113 / .2474 | 5/5 | .5816 / .2990 |
| 8 | 5/5 | .3884 / .3608 | 5/5 | .4938 / .3918 |
| 9 | 5/5 | .5005 / .3093 | 5/5 | .6111 / .2887 |
| 10 | 5/5 | .3881 / .2784 | 5/5 | .5017 / .3093 |
| 11 | 5/5 | .3476 / .2165 | 5/5 | .3941 / .2680 |
| 12 | 5/5 | .4153 / .2887 | 5/5 | .5161 / .3608 |

The SMD and KS limits were .10 and .20. Out-strength produced the largest
SMD in every seed and comparator; in-degree or out-strength usually produced
the largest KS, with real-state standard deviation doing so for seed 7 `M`.
The failure is therefore broad rather than one borderline control or one
unlucky seed.

Strict local CPU recovery authenticated the input seal, source archives,
checkpoint tensors and mutable-state snapshots; reconstructed every paired
coordinate; reproduced every control assignment, cost, overlap, balance
statistic and seed status; and reproduced the aggregate verdict. Cross-runtime
`log1p` reconstruction differed only in the last floating-point bit (maximum
observed absolute difference below `9e-16`). The local verifier permits
`rtol=0, atol=1e-12` only at that coordinate boundary; control identities,
costs, balance records and verdict remain exact. Recovery emitted
`ALPN_CALIBRATION_RECOVERY_VERIFIED`.

Per the prospective gate, the 68 fresh stories were never uploaded to the T4,
no fresh loss or accuracy was computed, no new `S`/`M` probe was fitted, and no
state intervention was scored. Consequently this experiment cannot adjudicate
ALPN specificity or causality. It establishes instead that the nominated ALPN
subsets lack adequate common support under the predeclared non-ALPN control
pool and coordinate set. The T4 was released after verified download, and the
authenticated Colab account reported no active sessions.
