# Central-brain TinyStories results

The central-brain wiring produced a consistent **within-run advantage over its
degree-preserving shuffled control at both tested sizes**. It did not beat the
optic-lobe-biased reference in absolute language quality.

All six conditions used one Tesla T4 session, seed 0, the same frozen
TinyStories corpus and 1,000-update budget. This is a controlled pilot result,
not statistical evidence that a biological brain region is specialized for
language.

## Main result

Lower NLL and perplexity are better. Accuracy is next-BPE-token accuracy on the
same 3,139 held-out test targets.

| Condition | Neurons | Directed pairs | Parameters | Test NLL | Test PPL | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual-biased real | 16,384 | 1,187,999 | 3,043,487 | **4.0448** | **57.10** | 27.97% |
| Visual-biased shuffled | 16,384 | 1,187,999 | 3,043,487 | 4.0541 | 57.64 | **28.32%** |
| Central N5,600 real | 5,600 | 1,187,928 | 3,032,632 | **4.4824** | **88.45** | **24.21%** |
| Central N5,600 shuffled | 5,600 | 1,187,928 | 3,032,632 | 4.8275 | 124.89 | 20.55% |
| Central N16,384 real | 16,384 | 3,222,739 | 5,078,227 | **4.7155** | **111.66** | **20.77%** |
| Central N16,384 shuffled | 16,384 | 3,222,739 | 5,078,227 | 5.2462 | 189.84 | 17.65% |

Every condition learned substantially from an initial perplexity near 4,096.
The paired real-versus-shuffled differences were:

| Graph | Real minus shuffled NLL | Real PPL reduction | Accuracy change |
| --- | ---: | ---: | ---: |
| Visual-biased N16,384 | -0.0093 | 0.93% | -0.35 percentage points |
| Central N5,600 | **-0.3451** | **29.18%** | **+3.66 percentage points** |
| Central N16,384 | **-0.5307** | **41.18%** | **+3.12 percentage points** |

The visual-biased real and shuffled conditions are effectively close under this
single run: NLL slightly favors real wiring while accuracy slightly favors the
shuffle. Both central real graphs improve NLL, perplexity and accuracy together
relative to their own shuffles.

This is the strongest evidence so far that anatomical arrangement can matter in
this model. It is still one seed and one custom split. The shuffle preserves
directed source/target degree multisets and parameter count, but it also breaks
the original pairing between contact-derived edge strengths and endpoints and
can create parallel edges or self-loops. The result therefore does not isolate
unweighted topology or establish a biological-language mechanism.

## What the central selections contain

Both central graphs are nested connected prefixes selected entirely inside the
32,164-neuron `cb_intrinsic` induced graph. No edge threshold was used: every
directed pair internal to each selected prefix retains its original integer
contact count.

The edge-matched N5,600 graph differs from the visual-biased reference by only
71 directed pairs. Available MaleCNS class annotations identify:

- all 4,064 annotated Kenyon cells;
- 338 dopaminergic neurons (`DAN`);
- 313 antennal-lobe projection neurons (`ALPN`);
- 97 mushroom-body output neurons (`MBON`);
- 90 antennal-lobe local neurons (`ALLN`);
- 25 central-complex-class neurons.

The N16,384 prefix retains the same 4,064 Kenyon cells, 340 DANs, 97 MBONs and
adds 1,674 central-complex-class neurons. Missing class labels remain
`cb_intrinsic`; they were not assigned a narrower class in the source table.

These labels make the N5,600 graph strongly mushroom-body-rich, but the model's
192 input and 256 readout ports are still seeded random nodes. They are not
anatomically identified sensory or output pathways.

## Scale did not improve this fixed-budget run

The larger central graph had the same neuron count as the visual-biased graph
but 2.71 times its directed pairs and 67% more trainable parameters. Under the
same 512,000-target budget, its real condition was worse than the smaller
central real condition:

- test perplexity 111.66 versus 88.45;
- accuracy 20.77% versus 24.21%;
- total condition runtime 214.21 versus 172.09 seconds.

This does not show that the added central neurons are harmful. The larger model
received fewer optimizer updates per parameter, has denser recurrent dynamics,
and was not retuned. A matched-update pilot deliberately cannot separate
undertraining from graph-scale effects.

## Runtime

Times include initial and final held-out evaluation, 1,000 updates, checkpoint
writes, generation and activation export. They are not training-only benchmark
times.

| Condition | Total seconds | Peak allocated CUDA tensors |
| --- | ---: | ---: |
| Visual-biased real | 138.11 | 290.76 MiB |
| Visual-biased shuffled | 146.57 | 297.21 MiB |
| Central N5,600 real | 172.09 | 248.01 MiB |
| Central N5,600 shuffled | 182.08 | 254.11 MiB |
| Central N16,384 real | 214.21 | 503.11 MiB |
| Central N16,384 shuffled | 281.64 | 522.35 MiB |

Runtime differences are descriptive; the run order was not randomized and each
condition was measured once.

## Generated continuations

All outputs include the prompt. The central real conditions learned recognizable
TinyStories fragments but still repeat and make grammatical or semantic errors.

Central N5,600 real, greedy:

```text
Once upon a time, a little girl named was a big tree. He was so excited and wanted to see the tree.
The little girl was a big tree and a big tree. She was so happy and the tree.
The little girl was a big tree and a big tree. He was so happy to see the tree.
```

Central N5,600 shuffled, greedy:

```text
Once upon a time.
"I'm you can can play with the tree.
The little girl was so so happy and to see the tree.
The little girl was so so happy and to see the tree.
The little girl was so so happy and to see the tree.
The little girl was so so happy and
```

Central N16,384 real, greedy:

```text
Once upon a time.
The little girl was so so excited.
The little girl was so so excited.
The little girl was so so excited.
The little girl was so excited.
The little girl was so excited.
The little girl was so excited to the tree.
The little girl was so excited.
```

Central N16,384 shuffled, greedy:

```text
Once upon a time.
The girl was so so excited and to the girl and was so so excited and to the girl and was so so excited and to the girl and was so so excited and to the girl and was so so excited and to the girl and was so so excited and to the girl and was so so excited and to
```

Full greedy and sampled outputs for all six conditions remain in each
`report.json` and in the verified recovery summary.

## Activation observations

Every condition exported the full state matrix before each of 64 greedy token
selections. The PNGs display only the 64 neurons ranked by mean absolute state;
that choice intentionally favors saturated-looking rows.

| Condition | Full cells with `abs(state) >= .99` | Displayed cells at threshold | Median neuron SD | Neurons with SD > .1 |
| --- | ---: | ---: | ---: | ---: |
| Visual-biased real | 7.80% | 94.09% | .0930 | 7,810 |
| Visual-biased shuffled | 7.98% | 94.04% | .1233 | 9,571 |
| Central N5,600 real | 67.98% | 95.31% | .0471 | 1,948 |
| Central N5,600 shuffled | 36.72% | 90.92% | .2681 | 5,110 |
| Central N16,384 real | 42.82% | 93.80% | .2830 | 10,552 |
| Central N16,384 shuffled | 76.46% | 94.90% | .0278 | 446 |

Saturation and temporal variation do not track language quality monotonically:
the better central N5,600 real condition varies less than its shuffle, while the
better central N16,384 real condition varies much more than its shuffle. These
are model-state descriptions, not causal explanations.

Each row below explicitly binds the otherwise identical image title to its
condition. Page 1 covers positions 1-32 and page 2 covers positions 33-64.

| Condition | Positions 1-32 | Positions 33-64 |
| --- | --- | --- |
| Central N5,600 real | [![Central N5600 real positions 1-32](results/central-brain-t4/central-n5600-real/activations-001.png)](results/central-brain-t4/central-n5600-real/activations-001.png) | [![Central N5600 real positions 33-64](results/central-brain-t4/central-n5600-real/activations-002.png)](results/central-brain-t4/central-n5600-real/activations-002.png) |
| Central N5,600 shuffled | [![Central N5600 shuffled positions 1-32](results/central-brain-t4/central-n5600-shuffled/activations-001.png)](results/central-brain-t4/central-n5600-shuffled/activations-001.png) | [![Central N5600 shuffled positions 33-64](results/central-brain-t4/central-n5600-shuffled/activations-002.png)](results/central-brain-t4/central-n5600-shuffled/activations-002.png) |
| Central N16,384 real | [![Central N16384 real positions 1-32](results/central-brain-t4/central-n16384-real/activations-001.png)](results/central-brain-t4/central-n16384-real/activations-001.png) | [![Central N16384 real positions 33-64](results/central-brain-t4/central-n16384-real/activations-002.png)](results/central-brain-t4/central-n16384-real/activations-002.png) |
| Central N16,384 shuffled | [![Central N16384 shuffled positions 1-32](results/central-brain-t4/central-n16384-shuffled/activations-001.png)](results/central-brain-t4/central-n16384-shuffled/activations-001.png) | [![Central N16384 shuffled positions 33-64](results/central-brain-t4/central-n16384-shuffled/activations-002.png)](results/central-brain-t4/central-n16384-shuffled/activations-002.png) |
| Visual-biased N16,384 real | [![Visual N16384 real positions 1-32](results/central-brain-t4/visual-n16384-real/activations-001.png)](results/central-brain-t4/visual-n16384-real/activations-001.png) | [![Visual N16384 real positions 33-64](results/central-brain-t4/visual-n16384-real/activations-002.png)](results/central-brain-t4/visual-n16384-real/activations-002.png) |
| Visual-biased N16,384 shuffled | [![Visual N16384 shuffled positions 1-32](results/central-brain-t4/visual-n16384-shuffled/activations-001.png)](results/central-brain-t4/visual-n16384-shuffled/activations-001.png) | [![Visual N16384 shuffled positions 33-64](results/central-brain-t4/visual-n16384-shuffled/activations-002.png)](results/central-brain-t4/visual-n16384-shuffled/activations-002.png) |

All 12 pages were opened after recovery. They have valid PNG signatures,
2424x2370 dimensions, continuous position ranges, readable original IDs,
input/readout markers, escaped BPE labels and a fixed signed -1 to 1 scale.

The states are bounded continuous model values. They are not firing rates,
attention, causal importance, measured biological activity, or evidence that a
negative state identifies an inhibitory neuron. Selected rows differ across
conditions, and row order is not a spatial brain map.

## Verification and artifacts

- Remote CUDA preflight: 71 tests passed on the actual T4.
- Local full suite on the final implementation: 236 passed, 54 CUDA skips.
- All six reports completed 1,000 updates without resume.
- Every checkpoint metadata record, parameter shape and finite numeric array
  passed local recovery validation.
- Every activation matrix has shape `64 x graph_nodes`, finite values, original
  node IDs and causal pre-selection contexts.
- All six independent condition ZIPs were downloaded and passed CRC checks
  before the remote session was released.
- The combined 263,043,627-byte result ZIP passed CRC and has SHA256
  `ea9695e363277f516ebbb1dc1331f8199b51ddd5adfdf9e3fd93fb51652ed37e`.
- `colab sessions` returned no active sessions after teardown.

The repository publishes the graph-selection manifest, result reports,
activation metadata, full activation arrays and PNG pages. Large optimizer
checkpoints and the combined ZIP remain local to avoid repository bloat.

The fixed setup and reproduction commands are in
[CENTRAL_BRAIN_PILOT.md](CENTRAL_BRAIN_PILOT.md).
