# Central-brain wiring replication

This protocol prospectively tests whether the seed-0 central N5,600
real-versus-shuffled difference survives fresh training randomness. It changes
no graph, model equation, port policy or optimization setting.

Seed 0 motivated this protocol and is excluded from the confirmatory decision.

## Frozen inputs

- Graph: `data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz`
- Graph SHA256:
  `738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935`
- Graph size: 5,600 neurons / 1,187,928 original directed pairs
- Corpus fingerprint:
  `fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575`
- Train-only byte BPE vocabulary 4,096
- Float32 anatomical recurrence; random 192-neuron inputs and 256-neuron
  readout
- Context 64, batch 8, 512 training targets per update
- Exactly 1,000 updates; AdamW learning rate .003, weight decay 0,
  gradient clipping 1
- Leak .5, initial gain .9, trainable sensory codes
- Final-update evaluation; no best-checkpoint selection
- 64-token generation and full-neuron activation export

## Condition matrix

Each seed is a matched real/shuffled pair. The same seed fixes sensory and
readout ports, token codes, readout initialization, edge-order initial values
and sampled training windows. Only shuffled target assignment differs.

| Fresh seed | First condition | Second condition |
| ---: | --- | --- |
| 1 | real | shuffled |
| 2 | shuffled | real |
| 3 | real | shuffled |
| 4 | shuffled | real |
| 5 | real | shuffled |
| 6 | shuffled | real |

All 12 conditions start independently. Learned parameters never transfer
between paired conditions. Infrastructure interruption may resume an exact
checkpoint, but failed or unfavorable seeds are never replaced.

The executor may inspect progress, device identity, finite losses and artifact
availability. It must not inspect or report held-out real-versus-shuffled
comparisons until all 12 conditions finish.

## Primary decision

For each fresh seed:

```text
delta = shuffled final test NLL - real final test NLL
```

Positive delta favors real wiring.

Advance to an anatomy-aware ALPN-to-MBON port experiment only when:

1. all six fresh deltas are positive; and
2. their median is at least .10 nat/token.

Six concordant signs give a two-sided exact sign-test probability of .03125
under a symmetric no-direction null. The .10 median is a prespecified practical
threshold, approximately a 9.5% paired perplexity reduction.

Mixed signs or ties are inconclusive. Six negative differences reverse the
proposed direction. Any complete result outside the advancement rule stops this
line without adding seeds or tuning hyperparameters.

Report every delta, mean, median, range, geometric-mean perplexity ratio,
accuracy difference, validation metrics, runtime and failure. Tokens inside a
story are not treated as independent experimental replicates.

## Scope

This is replication over initialization, random ports, training-window sampling
and target shuffling on one fixed custom corpus. It does not provide fresh-data
confirmation or isolate unweighted topology: shuffling also breaks the original
contact-strength/end-point pairing and may create parallel edges or self-loops.

Central N16,384, visual controls, anatomy-aware ports and saturation tuning are
excluded. The larger central graph was under a different parameter/edge budget.
Changing ports or gain before replication would replace the hypothesis rather
than test it.

## Result status

**ADVANCE.** The prespecified rule passed on 2026-09-12. All six fresh-seed
deltas were positive and their median was .6587 nat/token, well above the .10
threshold. Seed 0 was not included in this decision.

### Test-set paired results

| Seed | Real PPL | Shuffled PPL | Delta NLL | Real acc | Shuffled acc | Acc difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 93.56 | 171.62 | .6067 | .2284 | .1743 | .0542 |
| 2 | 89.63 | 178.22 | .6873 | .2249 | .1768 | .0481 |
| 3 | 100.47 | 188.64 | .6300 | .2236 | .1768 | .0468 |
| 4 | 75.14 | 167.47 | .8015 | .2478 | .1953 | .0526 |
| 5 | 82.82 | 182.67 | .7911 | .2255 | .1692 | .0564 |
| 6 | 94.98 | 125.35 | .2774 | .2236 | .2001 | .0236 |

- Median delta: .6587 nat/token
- Mean delta: .6323 nat/token
- Delta range: [.2774, .8015] nat/token
- Real/shuffled geometric-mean perplexity ratio: .5314, or a 46.9% reduction
  for real wiring
- Mean test accuracy: .2290 real versus .1821 shuffled; difference .0469
- Two-sided exact probability for six concordant signs: .03125

### Validation and runtime

`R/S` means real/shuffled within the same seed.

| Seed | Valid NLL R/S | Valid PPL R/S | Valid acc R/S | Runtime s R/S | Peak MiB R/S |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.7395 / 5.2834 | 114.38 / 197.05 | .2158 / .1646 | 96.6 / 94.6 | 248.0 / 254.1 |
| 2 | 4.6237 / 5.2922 | 101.87 / 198.77 | .2180 / .1646 | 93.7 / 95.9 | 248.0 / 254.1 |
| 3 | 4.8108 / 5.3814 | 122.83 / 217.32 | .2092 / .1578 | 94.7 / 96.3 | 248.0 / 254.1 |
| 4 | 4.4641 / 5.2248 | 86.84 / 185.83 | .2352 / .1850 | 92.3 / 97.2 | 248.0 / 254.1 |
| 5 | 4.5943 / 5.3302 | 98.91 / 206.49 | .2144 / .1531 | 94.5 / 95.1 | 248.0 / 254.1 |
| 6 | 4.7167 / 4.9458 | 111.80 / 140.58 | .2131 / .1847 | 94.5 / 97.6 | 248.0 / 254.1 |

All 12 conditions completed without resume or failure on a Tesla T4. Their
summed condition runtime was 1,143.0 seconds (19.0 minutes); the maximum
reported allocated CUDA memory was 254.1 MiB.

### Recovery and provenance

- Frozen input:
  `artifacts/central-replication-t4-input-final.zip`
  (`34dcdc7e8f491ef04724e8266e590bec1fe3c2bdff41cee794a76a91cce5e6b0`)
- The final local input was explicitly uploaded as the remote fixed path
  `/content/central-replication-t4-input.zip`; setup verified every inner hash.
- Frozen source SHA256:
  `af5e03b2ab32bf0a0b4f2b9841247e8b7f07a70b591f39f67de6519f6e97dcac`
- Frozen worker SHA256:
  `7d527e034db5a32b1381c28a49449921b97ee53f28cc6c5c36fe15bd2460ff7a`
- Combined result:
  `artifacts/central-replication-t4-results.zip`
  (`9ec18bdc4184834eb5b7472783265bd3b22e384a480c40bf1deb37e0a19b0d87`)
- Strict local recovery:
  `results/central-brain-replication-t4-final/recovery.json`

The six pair archives and combined archive passed CRC checks. Final recovery
independently matched the frozen protocol, graph, corpus, source and worker;
required every model, AdamW and restorable RNG tensor with exact shape, dtype,
finite values and update count; required a contiguous 1,000-update trace and
unresumed T4 runtime identity; bound all activation arrays, causal contexts,
ports, parameter fingerprints and PNG/SVG files to their checkpoints; and
recomputed the same aggregate decision as the remote worker. The T4 session
was then released and the server reported no active sessions.

The frozen worker completed this run uninterrupted. A post-report interruption
edge case in that frozen worker's reuse path was found only after launch and
did not trigger because every runtime records `resumed: false`; the local
orchestration copy now requires an explicit complete-archive marker before
reuse.

### Interpretation and remaining limits

This confirms that, under the frozen random-port N5,600 setup, original central
wiring consistently learns the TinyStories objective better than its
degree-preserving shuffled-target control. It clears the prespecified gate for
the anatomy-aware ALPN-to-MBON port experiment.

It does not show that the model beats a same-scale Transformer, generalizes to
new corpora or connectomes, or isolates topology from contact-strength/end-point
pairing. The six seeds share one graph and one custom corpus, and the shuffled
multigraph may contain parallel edges or self-loops. The exact sign probability
describes paired seed concordance; TinyStories tokens are not independent
replicates.
