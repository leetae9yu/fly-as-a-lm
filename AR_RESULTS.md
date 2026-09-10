# Autoregressive connectome language pilot

Measured on 2026-09-10 using Colab Free and a Tesla T4.

## Conclusion

A trainable character language model using actual anatomical connections learned
held-out text statistics. On the 16,384-neuron subset, anatomical recurrence
reached **45.21% next-character accuracy and 2.8057 bits per character**.
Free generation remains incoherent and repetitive.

This demonstrates a usable connectome-constrained language-model implementation,
not a fluent conversational model or language ability inherent in a fly brain.
The rewired control reached the same accuracy, so this pilot does not establish
an advantage from biological wiring. The 166,700-neuron full retained graph was
capacity-tested, not trained on natural text.

## Held-out comparison

All scores below use the same 1,024 final-target windows from the fresh test
slice. Lower negative log likelihood (NLL, natural logarithms) and bits per
character (BPC) are better. Accuracy is higher-is-better.

| Model | Correct / 1,024 | Accuracy | NLL | BPC |
| --- | ---: | ---: | ---: | ---: |
| Unigram | 188 | 18.36% | 3.023092 | 4.3614 |
| Bigram | 322 | 31.45% | 2.306429 | 3.3275 |
| Trigram | 420 | 41.02% | 1.944107 | **2.8048** |
| Anatomical, trained | 463 | **45.21%** | 1.944738 | 2.8057 |
| Rewired, trained | 463 | **45.21%** | 1.958365 | 2.8253 |
| Anatomical, frozen core | 295 | 28.81% | 2.545252 | 3.6720 |

The anatomical model improves accuracy over the trigram by 4.20 percentage
points, but does **not** improve its BPC: its NLL is slightly worse. Its BPC is
only 0.0197 below the rewired model, with identical accuracy. There is one
initialization per condition; these are descriptive results, not significance
claims.

| Neural condition | Train accuracy / BPC | Validation accuracy / BPC | Initial test BPC |
| --- | ---: | ---: | ---: |
| Anatomical | 47.85% / 2.5843 | 47.56% / 2.6306 | 5.5852 |
| Rewired | 48.34% / 2.5591 | 47.36% / 2.6431 | 5.5847 |
| Frozen core | 25.98% / 3.7806 | 30.57% / 3.6859 | 5.5852 |

The anatomical and rewired models each train 1,216,719 parameters, of which
12,336 belong to the readout. The frozen control trains only those 12,336
readout parameters. All three use 4,000 updates, seed 0, batch 8, context 32,
and the same private window-sampling stream: 1,024,000 training targets each.
Saved RNG states agree across conditions. No validation or test score selected
the budget or final checkpoint. Full protocol: `AUTOREGRESSIVE.md`.

## Does the recurrent core matter?

Disabling the recurrent operator reduces all three models to 18.36% test
accuracy. Ablated BPC is 15.1999 for anatomical, 14.2670 for rewired, and 4.5059
for frozen. The trained decoder is badly calibrated without its learned core.

This intervention removes input propagation through anatomical edges as well as
recurrent communication. It demonstrates dependence on the core computation,
not specifically on long-term memory. The frozen comparison provides separate
evidence that training the core improves prediction.

## Free generation

Unedited anatomical-model continuations from the training-only prompt:

```text
= valkyria chronicles iii = senj
```

Sampled continuation:

```text
el arment some e kordungsesy the have to reage poluches nov reperateen thas sere hed at roptrale hith relayp , semunn anced whe eimol smankay an fhelait cormecration , surcy nergo . " h werve under hir hule wor sernom nemy ; b1 a follen gs
```

Greedy continuation:

```text
an the rearor the reat reat reared the regorn the rear , and the rear the rearor the regorn the semper of the stare seral and the regorn the semper of the stare seral and the regorn the regorn the regorn the regorn the regorn the regorn the
```

These contain English-like character patterns and some words, but do not form
coherent sentences. Greedy generation enters repetitive loops. The complete
240-character strings, including trailing spaces, and both control conditions
are preserved in their JSON reports.

## T4 scale and throughput

MaleCNS v1.0 contains brain and ventral nerve cord. The full retained annotated
neuronal graph has 166,700 neurons, 25,582,938 directed pairs, and 124,177,617
contacts. The trained subset has 16,384 neurons and 1,187,999 pairs; it is an
optic-lobe-biased induced hub-BFS subset, not a whole brain.

Synthetic forward/backward/AdamW measurements used batch 8, context 32 and
65,536-edge gradient chunks. Times are means of two warm updates after one cold
update; these are short capacity probes, not sustained-throughput guarantees.

| Neurons | Directed pairs | Before cache, s/update | Cached, s/update | Cached peak allocation, GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 17,596 | 0.0688 | 0.0476 | 0.0679 |
| 4,096 | 82,836 | 0.0955 | 0.0520 | 0.0831 |
| 16,384 | 1,187,999 | 0.4979 | 0.1879 | 0.1973 |
| 166,700 | 25,582,938 | 11.8719 | 4.6262 | 2.5908 |

Caching reduced full-graph update time by about 2.57 times. Increasing the chunk
to 1,048,576 did not materially improve throughput, so training retained 65,536.
The full graph fits comfortably in T4 memory, but 4,000 updates at the measured
rate would take about 5.1 hours for one trainable condition alone. The subset was
chosen for a practical matched-control pilot. GPU allocation excludes driver
overhead; cached full-graph process peak RSS was 5,019,918,336 bytes.

Actual training-loop times were 769.5 seconds anatomical, 861.4 rewired, and
104.4 frozen. The three run calls, including their evaluation and checkpoint
work, totaled 30.38 minutes, excluding environment setup and process startup.
Measured pilot peak GPU allocations were 211,896,320, 222,596,608, and
161,189,376 bytes respectively.

Runtime: Torch 2.11.0+cu128, CUDA 12.8, NumPy 2.5.3, one CPU Torch thread,
float32 model parameters, and CUDA deterministic algorithms disabled.

## Checkpoint verification

Every condition restored model parameters, Adam moments, window RNG, update
count, and trace exactly. The stored sampled and greedy generations were
reproduced. Separate copies continued from update 4,000 to 4,004 without
modifying the published checkpoints.

After a save at 4,002, uninterrupted and reloaded copies each performed two
more updates. CUDA continuation was not bitwise identical:

| Condition | Max parameter difference | Max Adam difference | Final training NLL difference |
| --- | ---: | ---: | ---: |
| Anatomical | 9.239e-7 | 6.505e-8 | 1.192e-7 |
| Rewired | 1.192e-7 | 1.164e-9 | 0 |
| Frozen | 1.490e-8 | 9.313e-10 | 0 |

RNG states and update counts still matched. These small measured differences
are consistent with the documented CUDA sparse-reduction nondeterminism; do
not promise exact CUDA continuation. CPU verification was bitwise exact.

## Evidence and reproducibility

- `results/ar-main/seed-0/{real,shuffled,frozen}/`: final report, initial scores,
  training trace and checkpoint. Checkpoints are included in the delivery ZIP,
  not Git.
- `results/ar-resume/*/verification.json`: exact-restoration and continuation
  measurements, with original checkpoint hashes.
- `artifacts/ar-capacity-initial.json` and `artifacts/ar-capacity-cached.json`:
  original timing and memory observations.
- Graph NPZ SHA256:
  `b8e259a16c8cc8936c78cb906cf64bd816af81c22de5318029e3ba8d249e53c8`.
- Corpus NPZ SHA256:
  `ed6c27f74afb5dc8839b086c611914656c8d088519cbdd7f9de8c460ba4990b2`.
- Full-graph NPZ SHA256:
  `6f19085b11bf56e85e2129ca422cba3a4bdf00f5f580d7f0243e4fde359079d0`.

All downloaded checkpoint arrays were finite. Their configurations, graph and
corpus identities, 4,000-entry traces and update counts matched their reports.
Baseline records were identical across controls.

The 96-test suite, Ruff, fresh basedpyright, package build, CPU CLI/resume
surface, and CUDA dense-reference derivative checks passed. Bash syntax was
checked with `bash -n`; a Bash language server was not installed. PyTorch emitted
a sparse-invariant configuration warning, while the operator checks completed.
Upstream raw text whitespace was preserved rather than modifying source data
to satisfy Git's whitespace checker.

The Colab CLI's cached proxy connection expired after training. The existing
assignment was reattached with refreshed proxy information, without unassigning
the VM. All completed models were recovered before GPU verification. The
32,286,857-byte result archive passed ZIP CRC checks and matched remote SHA256
`8f9a94c804d84bb3bd48a8b29a81b1dd007adb065372810032ff6c6f6ab09d97`.
