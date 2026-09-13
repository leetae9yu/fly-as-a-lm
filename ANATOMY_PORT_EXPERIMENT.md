# Anatomy-aware ALPN-to-MBON port experiment

This prospective protocol tests whether anatomically identified input and
readout neurons improve TinyStories learning on the replicated central N5,600
graph, and whether any improvement is larger under original than shuffled
wiring. No result from seeds 0-6 enters its decision.

## Frozen anatomy

- Graph: `data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz`
- Graph SHA256:
  `738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935`
- Graph size: 5,600 neurons, 1,187,928 directed pairs, 4,507,454 contacts
- Annotation SHA256:
  `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`
- Exact publisher `class` labels inside the graph:
  - 313 antennal-lobe projection neurons (`ALPN`)
  - 4,064 Kenyon cells (`Kenyon_Cell`)
  - 97 mushroom-body output neurons (`MBON`)
- Port arrays: `data/central_connectome/anatomy_ports.npz`
- Port-array SHA256:
  `e9e230da2dbf8fa3c6bf2e10df571dc5519203518d196b3956e19a83429e4a78`

The induced graph retains 22,340 ALPN-to-Kenyon directed pairs with 389,441
contacts, 61,210 Kenyon-to-MBON pairs with 463,640 contacts, and 186 direct
ALPN-to-MBON pairs with 1,163 contacts. Every one of the 97 MBONs is reachable
from at least one ALPN by a directed path inside this graph.

Anatomy-aware conditions inject each token code into all 313 ALPNs and read
logits from all 97 MBONs. Random controls use 313 input and 97 readout neurons
sampled without replacement from nodes that are neither ALPN nor MBON. Random
input and readout sets are disjoint and are reused across all applicable
conditions within a seed.

## Six-condition matrix

Fresh seeds are 7-12. Each seed runs these six independent conditions:

| Name | Wiring | Input ports | Readout ports |
| --- | --- | --- | --- |
| `real-alpn_mbon` | original | all ALPN | all MBON |
| `real-alpn_random` | original | all ALPN | seeded random |
| `real-random_mbon` | original | seeded random | all MBON |
| `real-random_random` | original | seeded random | seeded random |
| `shuffled-alpn_mbon` | shuffled targets | all ALPN | all MBON |
| `shuffled-random_random` | shuffled targets | seeded random | seeded random |

The six base conditions are cyclically rotated once across the six seeds so
each occupies every execution position exactly once. All learned tensor shapes
are identical across conditions. Within a seed, code values, readout values,
edge-order initial weights, neuron biases, output biases and sampled training
windows start identically. Only explicit port indices and, for shuffled
conditions, target assignment differ.

The shuffled graph preserves source and target degree multisets and edge-order
initial values but breaks original endpoint pairing. It may contain multiedges
or self-loops.

## Frozen optimization

- Same train-only 4,096-token byte BPE corpus and fingerprint as the replication
- Context 64, batch 8: 512 next-token targets per update
- Exactly 1,000 updates; no best-checkpoint selection
- AdamW learning rate .003, weight decay 0, gradient clipping 1
- Leak .5, initial gain .9, trainable sensory codes
- Final held-out evaluation over every within-story target
- 64-token generation and full-neuron activation export

## Prospective contrasts

For seed `s`, let `L(wiring, policy)` be final test NLL.

Primary anatomy-port gain:

```text
G_s = L(real, random_random) - L(real, alpn_mbon)
```

Shuffled-port gain and wiring interaction:

```text
H_s = L(shuffled, random_random) - L(shuffled, alpn_mbon)
K_s = G_s - H_s
```

Positive `G_s` favors ALPN-to-MBON ports on original wiring. Positive `K_s`
means that port gain is larger under original than shuffled wiring.

The real-wiring 2x2 descriptive contrasts are:

```text
input_gain  = ((L(random_mbon) - L(alpn_mbon))
             + (L(random_random) - L(alpn_random))) / 2
output_gain = ((L(alpn_random) - L(alpn_mbon))
             + (L(random_random) - L(random_mbon))) / 2
interaction = L(alpn_random) + L(random_mbon)
            - L(alpn_mbon) - L(random_random)
```

Positive input/output values favor ALPN/MBON placement. Positive interaction is
superadditive NLL reduction. These three contrasts are descriptive and have no
advancement thresholds.

## Decision rule

The primary gate passes only when all six `G_s` values are strictly positive
and their median is at least .10 nat/token. The topology gate passes only when
all six `K_s` values are strictly positive and their median is at least .05.
Unrounded seedwise contrasts are evaluated before aggregation; threshold
equality passes and zero fails.

| Primary gate | Topology gate | Verdict |
| --- | --- | --- |
| pass | pass | `joint_success` |
| pass | fail | `primary_only` |
| fail | pass | `interaction_only` |
| fail | fail | `neither_gate_passed` |

Each endpoint also records `pass`, `positive_subthreshold`, `reversed`, or
`inconclusive`. Six concordant signs have a two-sided exact sign probability of
.03125 under independent equiprobable signs. The practical thresholds are not
null-hypothesis tests, and the dependent probabilities are not multiplied.

Only `joint_success` advances to a pathway-mechanism experiment.

## Sealing and recovery

Held-out metrics remain hidden until all 36 conditions finish. Metric-free
condition and seed completion markers may be observed, and each seed archive
may be downloaded and CRC-checked as it completes. Failed seeds are never
replaced; exact checkpoint resume is allowed only when every graph, corpus,
port, config and runtime identity matches.

Final local recovery must independently verify all source hashes, condition
names and orders, exact port indices, model/AdamW/RNG tensor state, contiguous
1,000-update traces, unresumed T4 runtime identity, causal activation arrays and
figures, and both aggregate decisions.

## Limits

This is one fixed graph and custom corpus. ALPN, Kenyon and MBON are publisher
class annotations, not a complete functional taxonomy. Random controls are
capacity-matched but not degree-matched, so a positive port effect includes
anatomical placement and graph-centrality differences. The six-condition design
cannot identify shuffled input/output main effects or a three-way
wiring-by-input-by-output interaction. It does not test whether the model beats
an ordinary Transformer.

## Result status

Completed on Tesla T4 on 2026-09-12. All 36 conditions passed strict local
recovery. The final archive is
`artifacts/anatomy-factorial-t4-results.zip` (SHA256
`42eb30e624ab043d29dc3fea02779717e6a12ba3796baaaeb31b43bb9c18cb01`);
the independent recovery record is
`results/anatomy-factorial-t4/recovery.json`.

### Verdict

**`neither_gate_passed`; do not advance to the pathway-mechanism experiment.**

The primary anatomy-port contrast reversed in every seed: using all ALPNs as
inputs and all MBONs as readout made held-out NLL worse than the capacity-matched
random/random ports.

| Seed | Primary gain | Topology interaction | ALPN input gain | MBON output gain | Input-output interaction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -.6347 | +.1556 | -.0282 | -.6064 | +.0395 |
| 8 | -.6630 | -.0691 | -.0297 | -.6333 | +.0545 |
| 9 | -.4805 | -.0181 | -.1712 | -.3094 | +.3219 |
| 10 | -.6569 | -.0043 | -.0241 | -.6328 | +.0409 |
| 11 | -.6163 | +.1170 | -.0278 | -.5885 | +.0438 |
| 12 | -.5184 | +.0611 | +.1846 | -.7030 | -.6628 |

Primary gain had median **-.6255 nat/token**, mean -.5950 and range
[-.6630, -.4805]. Because all six signs were negative, its endpoint is
`reversed` and its two-sided unanimous-sign probability is .03125. The
predeclared requirement was six positive values with median at least +.10.

Topology interaction had median **+.0284**, mean +.0404 and range
[-.0691, +.1556]. Only three of six values were positive, so its endpoint is
`inconclusive`; it fails both unanimity and the +.05 median threshold.

### Condition-level scale

| Wiring and ports | Mean test NLL | Geometric-mean PPL | Mean accuracy |
| --- | ---: | ---: | ---: |
| real / ALPN-MBON | 6.1925 | 489.07 | 7.14% |
| real / ALPN-random | 5.6001 | 270.45 | 12.91% |
| real / random-MBON | 6.1629 | 474.81 | 7.32% |
| real / random-random | 5.5975 | 269.76 | 13.05% |
| shuffled / ALPN-MBON | 6.1836 | 484.72 | 7.28% |
| shuffled / random-random | 5.5482 | 256.78 | 12.81% |

The descriptive ALPN input gain was positive in only one seed: median -.0280,
mean -.0161, range [-.1712, +.1846]. The MBON output gain was negative in all
six seeds: median -.6196, mean -.5789, range [-.7030, -.3094]. Therefore the
joint port failure is dominated by forcing the language decoder through the 97
MBON nodes, not by a stable disadvantage from injecting tokens into ALPNs.

Representative seed-7 and seed-12 activation heatmaps were visually inspected
after recovery. Titles, token axes, stable neuron IDs, input/readout markers and
the common [-1, 1] color scale rendered correctly. The selected high-activity
neurons mostly form nearly constant saturated red/blue horizontal bands with
little token-position variation. This is descriptive rather than a causal
diagnosis, but it is consistent with a saturated recurrent state offering a
poor 97-MBON language readout.

This result does not negate the earlier real-wiring advantage over shuffled
wiring. It says that the coarse publisher classes `ALPN` and `MBON` are not
drop-in language input/output ports for this recurrence and objective.

### Frozen regional-probe follow-up

The subsequent [frozen regional-probe experiment](REGIONAL_PROBE_EXPERIMENT.md)
used the twelve real/shuffled random-random checkpoints from this factorial
without retraining their recurrent parameters. Among three prospectively fixed
candidates, ALPN was the only region whose equal-width linear probes passed the
degree-matched, unigram, practical-effect and Holm gates. MBON probes remained
worse than degree-matched probes in all six real seeds.

That result identifies linear next-token accessibility from ALPN subsets, not a
successful ALPN-to-MBON pathway. Real-versus-shuffled ALPN differences were
mixed, so intact ALPN connectivity was not established as the cause.

### Recovery evidence

Local recovery independently required:

- the exact frozen graph, corpus, source, worker and port hashes;
- all six expected conditions for each of seeds 7-12;
- exact sensory/readout indices and manifest identity;
- complete model tensors, AdamW moments, restorable RNG and 1,000 contiguous
  trace entries;
- exact Tesla T4, Torch, CUDA, thread and execution-flag identities;
- causal full-neuron activation arrays and fully decodable PNG / well-formed
  SVG figures; and
- byte-identical remote and independently recomputed local summaries.

All 36 runtime records have `resumed=false`.

The first T4 session completed and packaged all 36 conditions but the Colab
assignment disappeared before seed 12 and the combined archive could be
downloaded. Seed 7-11 archives had already been downloaded and CRC-verified.
A replacement T4 restored those exact 30 condition archives under the same
frozen protocol and worker; the worker reused them and reran only the same
predeclared seed-12 conditions. No metric from the lost seed-12 attempt was
available, no substitute seed was introduced, and no held-out metric was read
until the replacement combined archive passed CRC and strict local recovery.
The replacement execution was held open until downloads were acknowledged,
then released. The Colab backend reported no active sessions afterward.

The unavailable seed-12 archive from the first assignment prevents checking
bitwise equality between its lost attempt and the retained rerun. This
infrastructure deviation should remain attached to the result even though it
could not create metric-based selection.
